#!/usr/bin/env python3
"""Contour testing scripts"""

from base_util import *
from contour_util import *
from vector_plot import *


from descartes import PolygonPatch


def test_linear():
    regions = numpy.zeros((20, 20))
    regions[1:-1, 1:-1] = 1
    regions[8:13, 8:] = 0
    regions[5:16, 5:16] = 0
    points = numpy.array(numpy.where(regions == 1)).T
    poly = Polygon(estimate_contour(points, 1))
    assert poly.is_valid
    assert (
        len(
            [
                tuple(elem)
                for elem in numpy.array(numpy.where(regions == 0)).T
                if poly.covers(Point(tuple(elem)))
            ]
        )
        < 4
    )
    assert (
        len(
            [
                tuple(elem)
                for elem in numpy.array(numpy.where(regions == 1)).T
                if not poly.covers(Point(tuple(elem)))
            ]
        )
        < 4
    )


file_name = "samples/cl-sample3.jpg"
image = load_image(file_name)
cfd_image = classify_eh_hybrid(image)
centroids = numpy.unique(cfd_image)
regions = segment(image, cfd_image)
polygons = seed_polygons(region_outlines(regions))


points = numpy.array(numpy.where(regions == 14)).T
points = numpy.array(numpy.where(region_outlines(regions) == 14)).T
fig, ax = pyplot.subplots()
ax.scatter(*zip(*points))
ax.add_patch(PolygonPatch(polygons[14]))
pyplot.show()
