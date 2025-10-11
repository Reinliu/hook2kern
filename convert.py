#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, argparse
from fractions import Fraction
from pathlib import Path

# 固定默认行为 
DEFAULT_OCTAVE_ANCHOR = 4
GRID_AUTO       = True   # 自动选择 2^N 网格（≤ 1/64）
GRID_DIV        = 8      # 仅当 GRID_AUTO=False 时使用
QUANTIZE_ONSET  = True   # onset 吸附网格
QUANTIZE_DUR    = True   # 时长吸附网格
FILL_CHORDS     = False  # 和弦不铺满，只在 onset 行显示
INCLUDE_METADATA= True   # 输出参考元数据

# helpers: pitch / key / duration 
PC_NAMES_SHARP = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# 常见和弦模式（根位相邻音程，单位：半音）
PATTERNS = {
    # --- Triads ---
    (4, 3):   ("",        "triad"),    # major
    (3, 4):   ("m",       "triad"),    # minor
    (3, 3):   ("dim",     "triad"),    # diminished
    (4, 4):   ("aug",     "triad"),    # augmented
    (5, 2):   ("sus4",    "triad"),    # sus4
    (2, 5):   ("sus2",    "triad"),    # sus2

    # --- 6th chords ---
    (4, 3, 2): ("6",      "6th"),      # major6  (1-3-5-6)
    (3, 4, 2): ("m6",     "6th"),      # minor6  (1-b3-5-6)
    (4, 3, 2, 4): ("6/9", "69"),       # major6/9
    (3, 4, 2, 4): ("m6/9","69"),       # minor6/9

    # --- 7th chords ---
    (4, 3, 3): ("7",      "7th"),      # dominant7 (1 3 5 b7)
    (3, 4, 3): ("m7",     "7th"),      # minor7    (1 b3 5 b7)
    (4, 3, 4): ("maj7",   "7th"),      # maj7      (1 3 5 7)
    (3, 3, 4): ("m7b5",   "7th"),      # half-diminished (ø7)
    (3, 3, 3): ("dim7",   "7th"),      # diminished seventh
    (4, 4, 3): ("maj7#5", "7th"),      # aug triad + maj7
    (4, 4, 2): ("6#5",    "7th"),      # 常见于增和弦带6

    # --- 9th chords (with 7th present) ---
    (4, 3, 3, 4): ("9",    "9th"),     # 1 3 5 b7 9
    (3, 4, 3, 4): ("m9",   "9th"),     # 1 b3 5 b7 9
    (4, 3, 4, 3): ("maj9", "9th"),     # 1 3 5 7 9

    # --- 11th chords (with 7th present) ---
    (4, 3, 3, 4, 3): ("11",    "11th"),    
    (3, 4, 3, 4, 3): ("m11",   "11th"),
    (4, 3, 4, 3, 3): ("maj11", "11th"),

    # --- 13th chords (with 7th present) ---
    (4, 3, 3, 4, 3, 4): ("13",    "13th"),
    (3, 4, 3, 4, 3, 4): ("m13",   "13th"),
    (4, 3, 4, 3, 3, 4): ("maj13", "13th"),

    # --- add chords (no 7th) ---
    (4, 3, 7): ("add9",   "add"),      # major add9
    (3, 4, 7): ("madd9",  "add"),      # minor add9
    (4, 3, 10): ("add11", "add"),
    (3, 4, 10): ("madd11","add"),
    (4, 3, 12): ("add13", "add"),      
    (3, 4, 12): ("madd13","add"),
}

# 根据时间估计bpm
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

    # 1) 打包、排序（按 beat），去重（相同 beat/time 只留一个）
    pts = sorted([(float(b), float(t)) for b, t in zip(beats, times)], key=lambda x: (x[0], x[1]))
    dedup = []
    last_b, last_t = None, None
    for b, t in pts:
        if last_b is None or abs(b - last_b) > 1e-9 or abs(t - last_t) > 1e-9:
            dedup.append((b, t))
            last_b, last_t = b, t
    pts = dedup

    # 2) 过滤非单调与奇异值（时间必须递增，拍必须递增）
    mono = []
    prev_b, prev_t = None, None
    for b, t in pts:
        if prev_b is None or (b > prev_b and t > prev_t):
            mono.append((b, t))
            prev_b, prev_t = b, t
    pts = mono
    if len(pts) < 2:
        return None

    # 3) 线性回归（最小二乘）：time = slope * beat + intercept
    n = len(pts)
    mean_b = sum(b for b, _ in pts) / n
    mean_t = sum(t for _, t in pts) / n
    num = sum((b - mean_b) * (t - mean_t) for b, t in pts)
    den = sum((b - mean_b) ** 2 for b, _ in pts)
    if den <= 1e-12:
        slope = None
    else:
        slope = num / den  # sec per beat

    bpm = None
    if slope and slope > 1e-6:
        bpm = 60.0 / slope

    # 4) 兜底：回退到相邻间隔中位数（剔除极端 dt）
    if bpm is None or not (0.1 <= bpm <= 1000):
        tempos = []
        for i in range(1, len(pts)):
            db = pts[i][0] - pts[i-1][0]
            dt = pts[i][1] - pts[i-1][1]
            if db > 0 and dt > 0.05 and dt < 5.0:  # 剔除近零/异常大间隔
                tempos.append(60.0 * db / dt)
        if tempos:
            tempos.sort()
            mid = len(tempos) // 2
            bpm = tempos[mid] if len(tempos) % 2 == 1 else 0.5 * (tempos[mid-1] + tempos[mid])

    if bpm is None or bpm <= 0:
        return None

    # 5) 归一到常见节拍范围（处理 double/half-time）
    while bpm > 240:
        bpm /= 2.0
    while bpm < 40:
        bpm *= 2.0

    return bpm


def midi_to_kern(midi: int) -> str:
    """MIDI → **kern 音名（不含时值）"""
    pc = int(midi) % 12
    octv = int(midi) // 12 - 1
    base = PC_NAMES_SHARP[pc]
    letter = base[0].lower()
    accidental = "#" if len(base) == 2 else ""
    if   octv > 4: marks = "'" * (octv - 4)
    elif octv < 4: marks = "," * (4 - octv)
    else:          marks = ""
    return f"{letter}{accidental}{marks}"

def key_token(tonic_pc: int, intervals: list) -> str:
    """非常简化的大小调判定"""
    major6 = [2,2,1,2,2,2]
    minor6 = [2,1,2,2,1,2]
    intervals = intervals or []
    mode_minor = intervals[:6] == minor6
    name = PC_NAMES_SHARP[int(tonic_pc) % 12]
    return f"*{name.lower()}:" if mode_minor else f"*{name}:"

# 将“拍”为单位的时长拆成拍长片段（1、1/2、1/4、…），返回 Fraction 列表
def split_beats_into_parts(beats: float):
    if beats is None or beats <= 1e-10:
        return []
    remaining = Fraction(beats).limit_denominator(256)
    allowed = [Fraction(2,1), Fraction(1,1), Fraction(1,2),
               Fraction(1,4), Fraction(1,8), Fraction(1,16), Fraction(1,32), Fraction(1,64)]
    out = []
    for val in allowed:
        while remaining >= val:
            out.append(val)
            remaining -= val
    # >1/128 拍的尾巴吸附到最近片段
    if remaining > 0 and remaining > Fraction(1,128):
        closest = min(allowed, key=lambda x: abs(x - remaining))
        out.append(closest)
    return out

def chord_symbol(root_pc: int, intervals: list, inversion: int) -> str:
    """和弦文本：识别常见性质 + 通用转位低音 root + sum(intervals[:inv])"""
    root_pc = int(root_pc) % 12
    key = tuple(int(x) for x in (intervals or []))
    inv = max(0, int(inversion or 0))
    quality, _ = PATTERNS.get(key, ("5", "other"))
    root = PC_NAMES_SHARP[root_pc]
    sym = f"{root}{quality}"
    if inv > 0 and inv <= len(key):
        bass_pc = (root_pc + sum(key[:inv])) % 12
        sym += f"/{PC_NAMES_SHARP[bass_pc]}"
    return sym

# ============================== grid / quantize ==============================

def _valid_note(n) -> bool:
    """旋律音合法性检查"""
    try:
        return (n is not None and
                float(n["offset"]) - float(n["onset"]) > 1e-9 and
                isinstance(n.get("pitch_class"), (int, float)) and
                isinstance(n.get("octave"), (int, float)))
    except Exception:
        return False

def _detect_min_step(melody, harmony):
    """粗测时轴最小相邻差（Fraction）"""
    vals = []
    def push(x):
        try:
            vals.append(Fraction(x).limit_denominator(256))
        except Exception:
            pass
    for n in melody or []:
        push(n.get("onset")); push(n.get("offset"))
    for h in harmony or []:
        push(h.get("onset")); push(h.get("offset"))
    vals = sorted(set(vals))
    if len(vals) < 2:
        return Fraction(1,8)
    diffs = [vals[i]-vals[i-1] for i in range(1,len(vals)) if vals[i]>vals[i-1]]
    if not diffs:
        return Fraction(1,8)
    # 用离散 GCD 近似
    g = diffs[0]
    for d in diffs[1:]:
        a = int(g * 256); b = int(d * 256)
        while b:
            a, b = b, a % b
        g = Fraction(a, 256)
    return g if g >= Fraction(1,64) else Fraction(1,64)

def _pick_power2_step(min_step_frac: Fraction) -> Fraction:
    """选择 <= min_step_frac 的最大 1/(2^N)，确保小节整除。"""
    candidates = [Fraction(1, 2**n) for n in range(0, 7)]  # 1, 1/2, ..., 1/64
    eligible = [c for c in candidates if c <= min_step_frac]
    return max(eligible) if eligible else Fraction(1, 64)

def plan_tied_segments(onset_beats_frac: Fraction, beat_parts, beat_unit: int):
    """
    输入拍长片段 -> 生成 [(起点拍, **kern分母, tie标记), ...]
    换算公式：拍长 p 与 **kern 分母 d 的关系： p = beat_unit / d  =>  d = beat_unit / p
    """
    segs, t = [], onset_beats_frac
    for i, p in enumerate(beat_parts):
        d = Fraction(beat_unit, 1) / p
        d_int = int(d) if d.denominator == 1 else int(round(float(d)))
        tie = "" if len(beat_parts) == 1 else ("[" if i == 0 else ("]" if i == len(beat_parts)-1 else "_"))
        segs.append((t, str(d_int), tie))
        t += p
    return segs

# ============================== core ==============================

def one_track_to_kern(track: dict, octave_anchor=DEFAULT_OCTAVE_ANCHOR):
    ann = track.get("annotations") or {}

    # —— 拍号
    if ann.get("meters"):
        m0 = ann["meters"][0]
        beats_per_bar = int(m0.get("beats_per_bar", 4))
        beat_unit     = int(m0.get("beat_unit", 4))
    else:
        beats_per_bar, beat_unit = 4, 4

    # —— 调号
    if ann.get("keys"):
        k0 = ann["keys"][0]
        k_token = key_token(k0.get("tonic_pitch_class", 0), k0.get("scale_degree_intervals", []))
    else:
        k_token = "*C:"

    # —— 旋律/和声（过滤非法）
    melody  = (ann.get("melody")  or [])
    harmony = (ann.get("harmony") or [])
    melody = [n for n in melody if _valid_note(n)]
    if not melody and not harmony:
        return ""

    # —— 选网格步长（Fraction）
    min_step = _detect_min_step(melody, harmony)
    step_frac = _pick_power2_step(min_step) if GRID_AUTO else Fraction(1, max(1, int(GRID_DIV)))

    # —— 估总长（拍）
    total_beats = Fraction(0,1)
    if melody:
        mb = max(Fraction.from_float(float(n["offset"])).limit_denominator(256) for n in melody)
        total_beats = max(total_beats, mb)
    if harmony:
        hb = max(Fraction.from_float(float(h.get("offset", 0.0))).limit_denominator(256) for h in harmony)
        total_beats = max(total_beats, hb)
    if total_beats <= 0:
        total_beats = Fraction(int(ann.get("num_beats") or 4), 1)

    # —— 生成行（Fraction 计时）
    rows, t = [], Fraction(0,1)
    while t <= total_beats + step_frac/1000:
        rows.append(t)
        t += step_frac
    if len(rows) < 2:
        rows.append(step_frac)
    grid_step = rows[1] - rows[0]

    def _snap_frac(x: float) -> Fraction:
        q = Fraction.from_float(float(x)).limit_denominator(256)
        k = int(round(float(q / grid_step)))
        return grid_step * k

    # —— 和弦 onset 映射到网格
    chord_on = {}
    for h in harmony:
        try:
            on = _snap_frac(h["onset"]) if QUANTIZE_ONSET else Fraction.from_float(float(h["onset"])).limit_denominator(256)
            chord_on[on] = h
        except Exception:
            continue

    # —— 头部 + 元数据
    lines = []
    if INCLUDE_METADATA:
        meta = track.get("hooktheory") or {}
        urls = (meta.get("urls") or {})
        tags = track.get("tags") or []
        mm_est = estimate_bpm(track.get('alignment') or {}) or 120
        lines.append(f"!!!ID:\t{meta.get('id') or ''}")
        lines.append(f"!!!Artist:\t{meta.get('artist') or ''}")
        lines.append(f"!!!Title:\t{meta.get('song') or ''}")
        lines.append(f"!!!HooktheorySongURL:\t{urls.get('song') or ''}")
        lines.append(f"!!!HookpadClipURL:\t{urls.get('clip') or ''}")
        lines.append(f"!!!YouTubeURL:\t{(track.get('youtube') or {}).get('url') or ''}")
        lines.append(f"!!!Tags:\t{', '.join(tags)}")
        lines.append(f"!!!Split:\t{track.get('split') or ''}")
        lines.append(f"!!!Swing:\t{(track.get('alignment') or {}).get('swing') or ''}")
        if mm_est:
            lines.append(f"!!!BPM_estimate:\t{int(round(mm_est))}")

    lines.append("**kern\t**harm")
    lines.append(f"*M{beats_per_bar}/{beat_unit}\t*")
    lines.append(f"{k_token}\t*")
    mm_token = f"*MM{int(round(mm_est))}"
    lines.append(f"{mm_token}\t*")

    # —— MIDI 计算
    def midi_for(n):
        octv = int(DEFAULT_OCTAVE_ANCHOR) + int(n["octave"])
        return 12 * (octv + 1) + int(n["pitch_class"])

    # —— 旋律：按拍分段 + 生成连音
    scheduled_notes = {}  # {row(Fraction): [token,...]}
    for n in melody:
        try:
            on_f  = Fraction.from_float(float(n["onset"])).limit_denominator(256)
            off_f = Fraction.from_float(float(n["offset"])).limit_denominator(256)
        except Exception:
            continue
        dur_f = off_f - on_f
        if dur_f <= 0:
            continue
        if QUANTIZE_ONSET: on_f = _snap_frac(float(on_f))
        if QUANTIZE_DUR:   dur_f = _snap_frac(float(dur_f))

        beat_parts = split_beats_into_parts(float(dur_f))
        segs = plan_tied_segments(on_f, beat_parts, beat_unit)
        kern_pitch = midi_to_kern(midi_for(n))
        for t_seg, d_str, tieflag in segs:
            t_row = _snap_frac(float(t_seg))
            tok = d_str + kern_pitch + tieflag
            scheduled_notes.setdefault(t_row, []).append(tok)

    # —— 小节线：tick 方式
    ticks_per_bar = int(Fraction(beats_per_bar,1) / grid_step)
    last_harm = "."

    # —— 逐行输出
    for r_i, b in enumerate(rows):
        if r_i != 0 and (r_i % ticks_per_bar == 0):
            bar_no = r_i // ticks_per_bar
            lines.append(f"={bar_no}\t=")

        mel_tok = scheduled_notes.get(b, ["."])[0]

        h = chord_on.get(b)
        if h is not None:
            harm_tok = chord_symbol(h.get("root_pitch_class", 0),
                                    h.get("root_position_intervals", []),
                                    h.get("inversion", 0))
            last_harm = harm_tok
        else:
            harm_tok = last_harm if FILL_CHORDS else "."

        lines.append(f"{mel_tok}\t{harm_tok}")

    lines.append("=||\t=||")
    lines.append("*-\t*-")
    return "\n".join(lines)

# ============================== I/O ==============================

def load_tracks(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        return data
    raise ValueError("Expected a dict keyed by track IDs at top level.")

def main():
    parser = argparse.ArgumentParser(
        description="Convert Hooktheory/SheetSage JSON → Humdrum **kern/**harm"
    )
    # 默认输出目录叫 'kern'（批量写出）
    parser.add_argument("--json_in", default="Hooktheory.json", help="输入 JSON（顶层以 ID 为键")
    parser.add_argument("--out", default="kern", help="输出目录(批量) 或 单个文件名(.krn)]")
    args = parser.parse_args()

    tracks = load_tracks(args.json_in)
    outpath = Path(args.out)

    # ---- 判定“目录输出/单文件输出”规则 ----
    # 1) 如果 out 以 .krn 结尾 => 单文件输出
    # 2) 否则 => 目录输出（不存在则自动创建）
    single_file = outpath.suffix.lower() == ".krn"

    if not single_file:
        # 目录批量输出（默认）
        outpath.mkdir(parents=True, exist_ok=True)
        written = 0
        for tid, track in tracks.items():
            text = one_track_to_kern(track)
            if not text.strip():
                print(f"Skipping {tid}: no usable melody/harmony.")
                continue
            (outpath / f"{tid}.krn").write_text(text, encoding="utf-8")
            written += 1
        print(f"Wrote {written} files to {outpath}")
        return

    # 单文件：JSON 多首时只取第一首
    tid, track = next(iter(tracks.items()))
    text = one_track_to_kern(track)
    if not text.strip():
        raise SystemExit(f"First track {tid} has no usable content; choose another or use a directory output.")
    outpath.write_text(text, encoding="utf-8")
    print(f"Wrote {outpath} for track {tid}")


if __name__ == "__main__":
    main()
