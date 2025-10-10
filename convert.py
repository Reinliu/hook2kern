#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, os, argparse
from fractions import Fraction
from pathlib import Path

# ==================== helpers: pitch / key / duration ====================
# 工具函数区：处理音名映射、调式标记、时值分解等等。

# 列出十二半音
PC_NAMES_SHARP = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# 根据beat和times估计bpm
def estimate_bpm(alignment: dict):

    if not alignment:
        return None
    src = alignment.get("refined") or alignment.get("user")
    if not src:
        return None

    beats = src.get("beats") or []
    times = src.get("times") or []
    if len(beats) < 2 or len(times) < 2 or len(beats) != len(times):
        return None

    bpms = []
    for i in range(1, len(beats)):
        db = float(beats[i]) - float(beats[i-1])
        dt = float(times[i]) - float(times[i-1])
        if dt > 0 and db > 0:
            bpms.append((db / dt) * 60.0)

    if not bpms:
        return None

    # 取中位数
    bpms.sort()
    mid = len(bpms) // 2
    if len(bpms) % 2 == 1:
        bpm = bpms[mid]
    else:
        bpm = 0.5 * (bpms[mid-1] + bpms[mid])

    return bpm


# 将MIDI音高号映射为 **kern 音名
def midi_to_kern(midi: int) -> str:

    pc = int(midi) % 12
    octv = int(midi) // 12 - 1  # MIDI 到 **kern 八度的换算
    base = PC_NAMES_SHARP[pc]   # 使用偏升号命名
    letter = base[0].lower()    # **kern 旋律音一般用小写
    accidental = "#" if len(base) == 2 else ""  # 这里只处理
    # 计算八度标记
    if   octv > 4: marks = "'" * (octv - 4)
    elif octv < 4: marks = "," * (4 - octv)
    else:          marks = ""
    return f"{letter}{accidental}{marks}"

# 由根音和间隔推断出大小调关系
def key_token(tonic_pc: int, intervals: list) -> str:
    major6 = [2,2,1,2,2,2]
    minor6 = [2,1,2,2,1,2]
    intervals = intervals or []
    mode_minor = intervals[:6] == minor6
    name = PC_NAMES_SHARP[int(tonic_pc) % 12]
    return f"*{name.lower()}:" if mode_minor else f"*{name}:"

# 将浮点拍子转成乐谱里的拍子
def beats_to_recip_frac(beats: float):

    if beats is None:
        return []
    if beats <= 1e-10:  # 非正时长直接跳过
        return []

    remaining = Fraction(beats).limit_denominator(128)
    allowed = [Fraction(1,1), Fraction(1,2), Fraction(1,4), Fraction(1,8), Fraction(1,16)]
    out = []

    # 贪心法覆盖：每次尽量取最大的允许时值片段
    for val in allowed:
        # 这里用 Fraction(0,1) 避免 0 作分母的异常
        while remaining - val >= Fraction(0,1) and remaining >= val:
            out.append(val)
            remaining -= val

    # 对剩余的“尾巴”做处理：若不是极小的数，就吸附到最近的允许值
    if remaining > Fraction(0,1):
        if remaining > Fraction(1,64):  # 小于等于 1/64 当作数值噪声忽略
            closest = min(allowed, key=lambda x: abs(x - remaining))
            out.append(closest)

    # 转换为 Humdrum 的分母字符串
    return [str(x.denominator) for x in out]

# 常见和弦的“根位相邻半音间隔”模式 → 文本标记
PATTERNS = {
    # --- Triads ---
    (4, 3): ("",      "triad"),   # 大三 C
    (3, 4): ("m",     "triad"),   # 小三 Cm
    (3, 3): ("dim",   "triad"),   # 减三 Cdim
    (4, 4): ("aug",   "triad"),   # 增三 Caug
    (5, 2): ("sus4",  "triad"),   # sus4 Csus4
    (2, 5): ("sus2",  "triad"),   # sus2 Csus2

    # --- Sixths (triad + 6th = 9 semitones above root) ---
    (4, 3, 2): ("6",   "6th"),    # 大六 C6 （根位：R-3-5-6 => 4,3,2）
    (3, 4, 2): ("m6",  "6th"),    # 小六 Cm6

    # --- Sevenths (triad + 7th) ---
    (4, 3, 3): ("7",      "7th"),     # 属七 C7
    (4, 3, 4): ("maj7",   "7th"),     # 大七 Cmaj7
    (3, 4, 3): ("m7",     "7th"),     # 小七 Cm7
    (3, 3, 4): ("m7b5",   "7th"),     # 半减七 Cø7
    (3, 3, 3): ("dim7",   "7th"),     # 全减七 C°7
    (3, 4, 4): ("m(maj7)","7th"),     # 小大七 Cm(maj7)（较少见）
    (4, 4, 3): ("maj7#5", "7th"),     # 大七增五 Cmaj7#5（较少见）
}

def chord_symbol(root_pc: int, intervals: list, inversion: int) -> str:

    root_pc = int(root_pc) % 12
    key = tuple(int(x) for x in (intervals or []))
    inv = max(0, int(inversion or 0))

    # 质量（quality）
    quality, _kind = PATTERNS.get(key, ("5", "other"))

    root = PC_NAMES_SHARP[root_pc]
    sym = f"{root}{quality}"

    # 通用转位低音（只要 inv 在长度范围内）
    if inv > 0 and inv <= len(key):
        bass_semitones = sum(key[:inv]) % 12
        bass_pc = (root_pc + bass_semitones) % 12
        sym += f"/{PC_NAMES_SHARP[bass_pc]}"

    return sym



# ============================== core ==============================
# 核心转换逻辑：从单个 track 的 JSON 注释生成 **kern/**harm 文本。


# 旋律音合理性检查
def _valid_note(n) -> bool:
    """
      - n 不为 None
      - offset > onset（时长为正）
      - pitch_class / octave 存在且为数值
    """
    try:
        return (n is not None and
                float(n["offset"]) - float(n["onset"]) > 1e-9 and
                isinstance(n.get("pitch_class"), (int, float)) and
                isinstance(n.get("octave"), (int, float)))
    except Exception:
        return False

def one_track_to_kern(track: dict, octave_anchor=4, grid_div=8,
                      # === 新增的稳健性与可读性开关（默认尽量对齐常见读谱） ===
                      quantize_onset=True,   # onset 吸附到网格，防止浮点抖动导致“错过行”
                      quantize_dur=True,     # duration 吸附到网格步长，连音更干净
                      scale_dur=True,        # 按拍值缩放：4/4 下 1 拍→'4'；3/8 下 1 拍→'8'
                      fill_chords=False):    # 和弦填充：没有新和弦时沿用上一个（替代 '.'）
    """
    将单个 track（含 annotations）转为 Humdrum 文本。
    关键步骤：
      1) 读 meter / key 并生成头部行
      2) 过滤旋律中的非法音（0 时长等）
      3) 计算总拍数与时间网格（grid_div 每拍的行数）
      4) 仅在旋律音 onset 行打印带时值的 **kern 记号；其他行用 '.'
      5) 和弦仅在 onset 行打印；其他行用 '.'
      6) 自动加小节线与收尾行
    """

    ann = track.get("annotations") or {}

    # --- 1) 拍号 ---
    if ann.get("meters"):
        m0 = ann["meters"][0]
        beats_per_bar = int(m0.get("beats_per_bar", 4))
        beat_unit     = int(m0.get("beat_unit", 4))
    else:
        beats_per_bar, beat_unit = 4, 4

    # --- 2) 调号（非常简化的大小调判断）---
    if ann.get("keys"):
        k0 = ann["keys"][0]
        k_token = key_token(k0.get("tonic_pitch_class", 0), k0.get("scale_degree_intervals", []))
    else:
        k_token = "*C:"

    # --- 3) 读旋律/和弦，并做 None 容错 + 旋律音过滤 ---
    melody  = (ann.get("melody")  or [])
    harmony = (ann.get("harmony") or [])
    melody = [n for n in melody if _valid_note(n)]  # 去掉 0 时长/缺字段等问题的音

    # 若旋律与和弦都为空，返回空串（调用方可选择跳过该 track）
    if not melody and not harmony:
        return ""

    # --- 4) 估计总拍数（用于落网格）---
    total_beats = 0.0
    if melody:
        total_beats = max(total_beats, max(float(n["offset"]) for n in melody))
    if harmony:
        total_beats = max(total_beats, max(float(h.get("offset", 0.0)) for h in harmony))
    if total_beats <= 0.0:
        total_beats = float(ann.get("num_beats") or 4.0)  # 兜底：如果都读不到，用 1 小节

    # --- 5) 时间网格：每拍切成 grid_div 份（默认 8，即八分拍网格）---
    step = Fraction(1, max(1, int(grid_div)))
    rows = []
    t = Fraction(0,1)
    end = Fraction(total_beats).limit_denominator(256)
    while t <= end + Fraction(1,1000):
        rows.append(float(t))
        t += step
    if len(rows) < 2:
        rows.append(float(step))  # 保证至少两行，避免后面计算步长时报错

    grid_step = float(rows[1] - rows[0])  # 每行增加的拍长
    def _snap(x: float, stepf: float) -> float:
        # 把 x 吸附到最近的网格点（避免 0.499999/0.333333 这类误差）
        return round(x / stepf) * stepf

    # --- 6) 建立和弦起始拍索引：只在 onset 行输出和弦 ---
    chord_on = {}
    for h in harmony:
        try:
            on = float(h["onset"])
            if quantize_onset:
                on = _snap(on, grid_step)  # 吸附到网格
            chord_on[round(on, 6)] = h
        except Exception:
            continue

    # --- 7) 开始拼 Humdrum 文本 ---
    lines = []
    lines.append("**kern\t**harm")                     # 两列：旋律与和声
    lines.append(f"*M{beats_per_bar}/{beat_unit}\t*") # 拍号
    lines.append(f"{k_token}\t*")                     # 调号

    # 计算bpm
    align = track.get("alignment") or {}
    bpm = estimate_bpm(align)
    mm_token = f"*MM{int(round(bpm))}" if bpm else "*MM120"
    lines.append(f"{mm_token}\t*")


    # Helper Function：按“相对八度 + 锚点”换算 MIDI
    def midi_for(n):
        octv = int(octave_anchor) + int(n["octave"])
        return 12 * (octv + 1) + int(n["pitch_class"])

    # 将旋律音按 onset 建立倒排索引，加速逐行查找
    starts = {}
    for n in melody:
        try:
            on = float(n["onset"])
            if quantize_onset:
                on = _snap(on, grid_step)  # 吸附到网格
            starts.setdefault(round(on, 6), []).append(n)
        except Exception:
            continue

    # 小节落点控制
    beats_per_bar_f = float(beats_per_bar)
    beat_in_bar = 0.0
    bar_no = 1

    # 和弦填充
    last_harm = "."

    emitted = 0

    # 逐行输出
    for r_i, b in enumerate(rows):
        # 小节线：除了第一行，每逢小节起点输出 "=N"
        if r_i != 0 and abs(beat_in_bar) < 1e-9:
            lines.append(f"={bar_no}\t=")
            bar_no += 1

        # 旋律列：只有 onset 行打印时值与音名，其余行 '.'
        mel_tok = "."
        onset_notes = starts.get(round(b, 6), [])
        if onset_notes:
            emitted += 1
            n = onset_notes[0]  # 单声部：取第一个即可

            # --- 时值（单位：拍）---
            dur_beats = float(n["offset"]) - float(n["onset"])
            if quantize_dur:
                dur_beats = _snap(dur_beats, grid_step)  # 吸附时值到网格步长（让 0.499999→0.5）

            # 让 4/4 下 1 拍→ '4'（四分），3/8 下 1 拍→ '8'（八分）等，更贴近常见记谱直觉
            dur_units = dur_beats * (4.0 / float(beat_unit)) if scale_dur else dur_beats

            parts = beats_to_recip_frac(dur_units)  # 将持续拍长拆分成若干二进制时值
            if parts:
                kern_pitch = midi_to_kern(midi_for(n))
                if len(parts) == 1:
                    mel_tok = parts[0] + kern_pitch
                else:
                    # 多段时值：在首段标记一个开 tie（'['），简化表达“还有后续”
                    mel_tok = parts[0] + kern_pitch + "["
            # 若 parts 为空（极短/0 时长），就保持 '.'

        # 和弦列：只在 onset 行打印和弦符号，其他行 '.'
        h = chord_on.get(round(b, 6))
        if h is not None:
            harm_tok = chord_symbol(h.get("root_pitch_class", 0),
                                    h.get("root_position_intervals", []),
                                    h.get("inversion", 0))
            last_harm = harm_tok
        else:
            harm_tok = last_harm if fill_chords else "."

        lines.append(f"{mel_tok}\t{harm_tok}")

        # 更新小节内拍计数；到小节末尾归零
        beat_in_bar += grid_step
        if beat_in_bar >= beats_per_bar_f - 1e-9:
            beat_in_bar = 0.0

    # 收尾小节线与结束标记
    lines.append("=||\t=||")
    lines.append("*-\t*-")

    return "\n".join(lines)

# ============================== I/O ==============================
# 装载与入口：仅使用普通 open 读取 .json；--out 既可指向文件，也可指向已有目录。

def load_tracks(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        return data
    raise ValueError("Expected a dict keyed by track IDs at top level.")

def main():
    """
    命令行入口：
      - 位置参数：json_in（输入 JSON）
      - --out：若为已有目录，则为每个 track 生成一个 .krn；否则写为单个 .krn（多 track 取第一个）
      - --octave-anchor：旋律相对八度的锚定八度（默认 4）
      - --grid-div：每拍划分的行数（默认 8 → 八分拍网格）
    """
    ap = argparse.ArgumentParser(description="Convert SheetSage/Hooktheory-style JSON → Humdrum **kern/**harm.")
    ap.add_argument("--json_in", default= 'Hooktheory.json', help="Input JSON keyed by track ID")
    ap.add_argument("--out", required=True, help="Output .krn file OR an existing directory (one file per ID)")
    ap.add_argument("--octave-anchor", type=int, default=4, help="Anchor for relative melody octaves (default 4)")
    ap.add_argument("--grid-div", type=int, default=8, help="Rows per beat (8 ⇒ 1/8-beat grid)")
    ap.add_argument("--quantize-onset",  action="store_true", default=True, help="onset 吸附到网格")
    ap.add_argument("--no-quantize-onset", dest="quantize_onset", action="store_false")
    ap.add_argument("--quantize-dur",    action="store_true", default=True, help="duration 吸附到网格")
    ap.add_argument("--no-quantize-dur", dest="quantize_dur", action="store_false")
    ap.add_argument("--scale-dur",       action="store_true", default=True, help="按拍值缩放")
    ap.add_argument("--no-scale-dur",    dest="scale_dur", action="store_false")
    ap.add_argument("--fill-chords",     action="store_true", default=False, help="和弦填充")
    args = ap.parse_args()

    tracks = load_tracks(args.json_in)
    outpath = Path(args.out)

    # 若 --out 指向“已存在的目录”，则批量输出
    if outpath.exists() and outpath.is_dir():
        written = 0
        for tid, track in tracks.items():
            text = one_track_to_kern(track,
                                     octave_anchor=args.octave_anchor,
                                     grid_div=args.grid_div,
                                     quantize_onset=args.quantize_onset,
                                     quantize_dur=args.quantize_dur,
                                     scale_dur=args.scale_dur,
                                     fill_chords=args.fill_chords)
            if not text.strip():
                # 空内容（无旋律/和声）则跳过
                print(f"Skipping {tid}: no usable melody/harmony.")
                continue
            (outpath / f"{tid}.krn").write_text(text, encoding="utf-8")
            written += 1
        print(f"Wrote {written} files to {outpath}")
        return

    # 否则写单文件（若 JSON 有多首，取第一首）
    tid, track = next(iter(tracks.items()))
    text = one_track_to_kern(track,
                             octave_anchor=args.octave_anchor,
                             grid_div=args.grid_div,
                             quantize_onset=args.quantize_onset,
                             quantize_dur=args.quantize_dur,
                             scale_dur=args.scale_dur,
                             fill_chords=args.fill_chords)
    if not text.strip():
        raise SystemExit(f"First track {tid} has no usable content; choose another or write to a directory.")
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"Wrote {args.out} for track {tid}")

if __name__ == "__main__":
    main()
