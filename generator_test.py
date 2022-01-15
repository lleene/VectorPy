#!/usr/bin/env python3
"""SVG generator testing scripts"""
from sys import exit
from os.path import basename
from glob import glob
from generator_util import *
from vector_plot import *

# open_in_window([dump_to_data("./samples/pepe.webp")])
dump_to_svg_file("./samples/pepe.webp","temp.svg")
exit(0)

for file_name in glob("./samples/*.jpg") + glob("./samples/*.png"):
    # print(f"{os.path.basename(file_name)[:-4]}.dn3.svg")
    dump_to_svg_file(file_name, f"./temp/{basename(file_name)[:-4]}.dn3.svg")
