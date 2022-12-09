#!/usr/bin/env python

import numpy as np
import cairo


with cairo.SVGSurface("test.svg", 100, 100) as surface:
    ctx = cairo.Context(surface)

    pattern = cairo.MeshPattern()
    pattern.begin_patch()
    pattern.move_to(0, 0)
    pattern.line_to(100, 0)
    pattern.line_to(100, 100)
    pattern.line_to(0, 100)
    pattern.set_corner_color_rgb(0, 1.0, 0.0, 0.0)
    pattern.set_corner_color_rgb(1, 0.0, 1.0, 0.0)
    pattern.set_corner_color_rgb(2, 0.0, 0.0, 1.0)
    pattern.set_corner_color_rgb(3, 1.0, 1.0, 0.0)
    pattern.end_patch()
    pattern.set_filter(cairo.FILTER_BEST)

    ctx.set_source(pattern)
    ctx.rectangle(0, 0, 100, 100)
    ctx.paint()


    surface.finish()


