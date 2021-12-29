"""Image Processing Utility Procedures"""

from typing import List
import numpy
from scipy.ndimage.filters import maximum_filter, median_filter
from scipy.ndimage.interpolation import shift
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


def matching_neighbours(image, cfd_image, threshold):
    return numpy.concatenate(
        [
            numpy.where(
                numpy.logical_or(
                    numpy.sum(
                        abs(image - shift(image, (ox, oy, 0), cval=0)), axis=2
                    )
                    <= threshold,
                    cfd_image - shift(cfd_image, (ox, oy), cval=0) == 0,
                ),
                True,
                False,
            )[:, :, None]
            for ox, oy in [(1, 1), (1, 0), (1, -1), (0, 1)]
        ],
        axis=2,
    )


def segment(image, cfd_image):
    regions = numpy.zeros(cfd_image.shape, dtype=int)
    match = matching_neighbours(image, cfd_image, 1)
    offsets = numpy.array([(1, 1), (1, 0), (1, -1), (0, 1)])
    region_counter = 0
    aliases = {}
    for x, y in numpy.ndindex(cfd_image.shape):
        xy_match = offsets[numpy.where(match[x, y, :])[0]]
        if xy_match.any():
            matched_regions = {regions[x - ox, y - oy] for ox, oy in xy_match}
            value = min(matched_regions)
            regions[x, y] = (
                unravel(value, aliases) if value in aliases else value
            )
            if len(matched_regions) > 1:
                aliases.update(
                    {elem: regions[x, y] for elem in matched_regions}
                )
        else:
            region_counter += 1
            regions[x, y] = region_counter
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
