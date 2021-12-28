"""Image Processing Utility Procedures"""

from typing import List
import numpy
from scipy.ndimage.filters import maximum_filter, median_filter
import cv2
import sys


def select_channel(data):
    padding = numpy.ones(data[:, :, None].shape) * 128
    return numpy.concatenate((data[:, :, None], padding, padding), axis=2)


def is_monochrome(image) -> bool:
    return image[:, :, 1:].std() < 5.0


def load_image(file_name: str):
    img = cv2.imread(file_name)
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)


def find_peaks_nd(data, size=5):
    return numpy.where(
        numpy.logical_and(data == maximum_filter(data, size=size), data > 0.0)
    )


def cluster_image(image, centroids):
    data = image.reshape(image.shape[0] * image.shape[1], *image.shape[2:])
    return (
        numpy.argmin(
            sum(
                (data[:, index] - centroids[index][:, numpy.newaxis]) ** 2
                for index in range(image.shape[2])
            ),
            axis=0,
        ).reshape(image.shape[0], image.shape[1])
        + 1
    )


def histogram_centroids(data):
    hcount, _ = numpy.histogramdd(
        data.reshape(data.shape[0] * data.shape[1], *data.shape[2:]),
        bins=numpy.repeat(
            numpy.arange(256)[:, None], data.shape[2], axis=1
        ).transpose(),
        density=True,
    )
    size = 32
    for step in range(4):  # do binary search here
        centroids = find_peaks_nd(hcount, size)
        size = (
            size + 16 / (1 << step)
            if len(centroids[1]) > 20
            else size - 16 / (1 << step)
        )
    return centroids


def partion_image(data):
    centroids = numpy.array(histogram_centroids(data))
    return cluster_image(data, centroids), centroids


def unravel(value, aliases):
    if aliases[value] in aliases and aliases[aliases[value]] < aliases[value]:
        aliases[value] = unravel(aliases[value], aliases)
    return aliases[value]


def segment(image, cfd_image):
    regions = numpy.zeros(cfd_image.shape, dtype=int)
    region_counter = 0
    aliases = {}
    for x, y in numpy.ndindex(cfd_image.shape):
        match = {
            int(regions[x + ox, y + oy])
            for ox, oy in [[0, -1], [-1, 0], [-1, -1], [-1, +1]]
            if 0 <= x + ox < regions.shape[0]
            and 0 <= y + oy < regions.shape[1]
            and (
                cfd_image[x, y] == cfd_image[x + ox, y + oy]
                or numpy.sum(abs(image[x, y] - image[x + ox, y + oy])) < 2.0
            )
        }
        if len(match) > 0:
            regions[x, y] = min(
                [unravel(elem, aliases) for elem in match if elem in aliases]
                or list(match)
            )
            if len(match) > 1:
                aliases.update({elem: regions[x, y] for elem in match})
        else:
            region_counter += 1
            regions[x, y] = region_counter
        for fx, fy in [
            (x + ox, y + oy)
            for ox, oy in [[0, +1], [+1, 0], [+1, -1], [+1, +1]]
            if 0 <= x + ox < regions.shape[0]
            and 0 <= y + oy < regions.shape[1]
            and (
                cfd_image[x, y] == cfd_image[x + ox, y + oy]
                or numpy.sum(abs(image[x, y] - image[x + ox, y + oy])) < 2.0
            )
        ]:
            regions[fx, fy] = regions[x, y]
    for x, y in numpy.ndindex(regions.shape):
        if regions[x, y] in aliases:
            regions[x, y] = unravel(regions[x, y], aliases)
    rlist, rcnts = numpy.unique(regions, return_counts=True)
    blist = rlist[numpy.where(rcnts < 50)]
    regions = numpy.where(numpy.isin(regions, blist), 0, regions)
    return regions


def find_edges(image, cfd_image):
    # conclusion threshold using numpy.linalg.norm() < 120 is
    # sufficient to differentiate soft/hard edges on first pass
    return numpy.fromiter(
        (
            numpy.linalg.norm(image[x, y, :] - image[x + ox, y + oy, :])
            for ox, oy in [[0, 1], [1, 0], [1, 1], [-1, 1]]
            for x, y in numpy.ndindex(cfd_image.shape)
            if 0 <= x + ox < cfd_image.shape[0]
            and 0 <= y + oy < cfd_image.shape[1]
            and cfd_image[x, y] != cfd_image[x + ox, y + oy]
        ),
        dtype=float,
    )


# =]
