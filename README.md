# Hook to kern conversion

### Environments:
This repo doesn't depend on any libraries. You should be able to use any python environments. (In my case it's Python 3.6.5)

### Step 1: Extract the json file from the compressed file. 

~~~
gzip -d Hooktheory.json
~~~


### Step 2: Convert the json file into kern.

~~~
python convert.py --json_in Hooktheory.json --out kern
~~~


Key steps:

1) Read meter/key and generate header lines

2) Filter out invalid melody notes (e.g., zero duration)

3) Compute total beats and build the time grid (grid_div rows per beat)

4) Print a **kern token with duration only at melody onsets; use '.' on other rows

5) Print harmony only at chord onsets; use '.' on other rows

6) Automatically add barlines and closing markers
