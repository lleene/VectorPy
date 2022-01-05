"""Image Processing Utility Procedures"""

from typing import List, Tuple
from tqdm import tqdm
import numpy
import cv2

# TODO implement numpy native method
from scipy.ndimage.filters import maximum_filter
from sklearn.cluster import KMeans


def is_monochrome(image) -> bool:
    return image[:, :, 1:].std() < 5.0


def derivative_features(image, size: int = 3):
    border = cv2.borderInterpolate(0, 1, cv2.BORDER_CONSTANT)
    return numpy.linalg.norm(
        numpy.concatenate(
            [
                cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=size, borderType=border),
                cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=size, borderType=border),
            ],
            axis=2,
        ),
        axis=2,
    ).astype(int)


def load_image(file_name: str, denoise_factor: int = None):
    img = cv2.imread(file_name)
    return cv2.cvtColor(
        cv2.fastNlMeansDenoisingColored(img, h=denoise_factor, hColor=denoise_factor)
        if denoise_factor
        else img,
        cv2.COLOR_BGR2RGB,
    )


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


def binary_search_histogram(color_samples):
    hcount, _ = numpy.histogramdd(
        color_samples,
        bins=numpy.repeat(
            numpy.arange(256)[:, None], color_samples.shape[1:], axis=1
        ).T,
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


def classify_eh_hybrid(data):
    kmeans = KMeans(init="random", max_iter=500, n_clusters=2, random_state=1337)
    kmeans.fit(numpy.ravel(derivative_features(data, 5))[:, None])
    noedge_colors = data.reshape(data.shape[0] * data.shape[1], 3)[
        numpy.where(kmeans.labels_ != kmeans.cluster_centers_.argmin())[0], :
    ]
    centroids = binary_search_histogram(noedge_colors)
    labeled_colors = (
        numpy.argmin(
            numpy.sum(
                (
                    data.reshape(data.shape[0] * data.shape[1], 3)[:, :, None]
                    - numpy.array(centroids)[None, :, :]
                )
                ** 2,
                axis=1,
            ),
            axis=1,
        )
        + 2
    )
    return numpy.where(
        kmeans.labels_ != kmeans.cluster_centers_.argmin(),
        1,
        labeled_colors,
    ).reshape(data.shape[:2])


def unravel_alias(value, aliases):
    if aliases[value] in aliases and aliases[aliases[value]] < aliases[value]:
        aliases[value] = unravel_alias(aliases[value], aliases)
    return aliases[value]


def shift_image(image, shift, cval):
    result = numpy.empty_like(image)
    if shift == (1, 0):
        result[:1, :] = cval
        result[1:, :] = image[:-1, :]
    if shift == (0, 1):
        result[:, :1] = cval
        result[:, 1:] = image[:, :-1]
    if shift == (-1, 0):
        result[-1:, :] = cval
        result[:-1, :] = image[1:, :]
    if shift == (0, -1):
        result[:, -1:] = cval
        result[:, :-1] = image[:, 1:]
    return result


def matching_classes(
    cfd_image: numpy.array, offsets: List[Tuple[int, int]] = [(1, 0), (0, 1)]
):
    return numpy.concatenate(
        [
            numpy.where(
                cfd_image - shift_image(cfd_image, (ox, oy), cval=0) == 0,
                True,
                False,
            )[:, :, None]
            for ox, oy in offsets
        ],
        axis=2,
    )


def matching_neighbours(
    image: numpy.array,
    cfd_image: numpy.array,
    threshold: int,
    offsets: List[Tuple[int, int]] = [(1, 0), (0, 1)],
):
    not_edge = numpy.where(cfd_image[:, :, None] != 1, image, (-1, -1, -1))
    return numpy.concatenate(
        [
            numpy.where(
                numpy.logical_or(
                    numpy.sum(
                        abs(image - shift_image(not_edge, (ox, oy), cval=0)),
                        axis=2,
                    )
                    <= threshold,
                    cfd_image - shift_image(cfd_image, (ox, oy), cval=0) == 0,
                ),
                True,
                False,
            )[:, :, None]
            for ox, oy in offsets
        ],
        axis=2,
    )


def segment(image, cfd_image, threshold=1):
    regions = numpy.zeros(cfd_image.shape, dtype=int)
    offsets = numpy.array([(1, 0), (0, 1)])
    match = matching_neighbours(image, cfd_image, threshold, list(offsets))
    region_counter = 1
    aliases = {}
    for x in tqdm(range(cfd_image.shape[0]), desc="Segmenting"):
        for y in range(cfd_image.shape[1]):
            if cfd_image[x, y] == 1:
                regions[x, y] = 1
                continue
            xy_match = offsets[numpy.where(match[x, y, :])[0]]
            if xy_match.any():
                matched_regions = {regions[x - ox, y - oy] for ox, oy in xy_match}
                value = min(matched_regions)
                regions[x, y] = (
                    unravel_alias(value, aliases) if value in aliases else value
                )
                if len(matched_regions) > 1:
                    aliases.update({elem: regions[x, y] for elem in matched_regions})
            else:
                region_counter += 1
                regions[x, y] = region_counter
    for x, y in numpy.ndindex(regions.shape):
        if regions[x, y] in aliases:
            regions[x, y] = unravel_alias(regions[x, y], aliases)
    rlist, rcnts = numpy.unique(regions, return_counts=True)
    blist = rlist[numpy.where(rcnts < 50)]
    regions = numpy.where(numpy.isin(regions, blist), 0, regions)
    return regions


# =]
