#!/usr/bin/env python3
"""Testing scripts"""

from vector_util import *
from vector_plot import *


show_segmentation("cl-sample2.jpg")

# shapes = {elem: numpy.where(regions == elem) for elem in numpy.unique(regions)}


class Shape:
    def __init__(self, pixel_index, pixe_color, pixel_class):
        self._points = pixel_index


# =]
