# Hook to kern conversion

### Step 1: Extract the json file from the compressed file. 

~~~
gzip -d Hooktheory.json
~~~


### Step 2: Convert the json file into kern.

~~~
python convert.py
~~~


So far this script separates the kern grids in 1/8 tempo, meaning that there are 8 grids for each tempo. 

