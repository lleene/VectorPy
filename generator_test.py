#!/usr/bin/env python3
"""SVG generator testing scripts"""

from generator_util import *
from vector_plot import *

file_name = "./samples/cl-sample3.jpg"
# dump_to_svg_file(file_name)
open_in_window([dump_to_data(file_name)])