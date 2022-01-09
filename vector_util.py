"""Image Processing Utility Procedures"""

from typing import List, Tuple, Union
from tqdm import tqdm
import numpy
import cv2

# TODO implement numpy native method
from scipy.ndimage.filters import maximum_filter
from sklearn.cluster import KMeans


def is_monochrome(image: numpy.array) -> bool:
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
    size = 50
    for step in range(4):  # do binary search here
        centroids = find_peaks_nd(hcount, size)
        size = (
            size + (32 / (1 << step))
            if len(centroids[1]) > 20
            else size - (32 / (1 << step))
        )
    return centroids


def classify_eh_hybrid(data):
    kmeans = KMeans(init="random", max_iter=100, n_clusters=2, random_state=1337)
    kmeans.fit(numpy.ravel(derivative_features(data, 3))[:, None])
    edge_class = kmeans.cluster_centers_.argmax()
    noedge_colors = data.reshape(data.shape[0] * data.shape[1], 3)[
        numpy.where(kmeans.labels_ != edge_class)[0], :
    ]
    centroids = binary_search_histogram(noedge_colors)
    labeled_colors = numpy.empty_like(data[:, :, 0])
    for row in numpy.ndindex(labeled_colors.shape[0]):
        labeled_colors[row, :] = (
            numpy.argmin(
                numpy.sum(
                    (data[row, :, :, None] - numpy.array(centroids)[None, None, :, :])
                    ** 2,
                    axis=2,
                ),
                axis=2,
            )
            + 2
        )
    return numpy.where(
        (kmeans.labels_ != edge_class).reshape(data.shape[:2]), labeled_colors, 1
    )


def unravel_alias(value, aliases):
    if aliases[value] in aliases and aliases[aliases[value]] < aliases[value]:
        aliases[value] = unravel_alias(aliases[value], aliases)
    return aliases[value]


def shift_image(image: numpy.array, shift: int, cval: Union[float, int]):
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


def morphic_filter(regions, minimum_count: int = 50):
    region_ids, region_counts = numpy.unique(regions, return_counts=True)
    region_filter = numpy.array([], dtype=int)
    for region_id in region_ids[numpy.where(region_counts >= minimum_count)]:
        points = numpy.where(regions == region_id)
        if numpy.unique(points[0]).size > 1 and numpy.unique(points[1]).size > 1:
            region_filter = numpy.insert(region_filter, 0, region_id)
    return numpy.where(numpy.isin(regions, region_filter), regions, 0)


def segment(image: numpy.array, cfd_image: numpy.array, threshold: int = 2):
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
    return morphic_filter(regions)


# =]
