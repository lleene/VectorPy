"""Image Processing Utility Procedures"""

from typing import List
import numpy
from scipy.ndimage.filters import maximum_filter, minimum_filter, median_filter
import cv2
from hilbert import decode


def select_channel(data):
    padding = numpy.ones(data[:, :, None].shape) * 128
    return numpy.concatenate((data[:, :, None], padding, padding), axis=2)


def image_classfied(data, cluster_labels: List[int]):
    bits = round(numpy.ceil(numpy.log2(len(cluster_labels)))) + 1
    color_map = numpy.concatenate(
        (
            ((decode(cluster_labels, 2, bits) + 0.5) * 256 / (bits)),
            numpy.ones((len(cluster_labels), 1)) * 128,
        ),
        axis=1,
    ).astype(numpy.uint8)

    def map_fnct(x):
        return color_map[x]

    return numpy.vectorize(map_fnct, signature="(n)->(n,3)")(data)


def is_monochrome(image) -> bool:
    return image[:, :, 1:].std() < 5.0


def load_image(file_name: str):
    img = cv2.imread(file_name)
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)


def find_peaks_nd(data, size=5):
    return numpy.where(
        numpy.logical_and(data == maximum_filter(data, size=size), data > 0.0)
    )


def denoise_class(data, size: int = 3):
    return median_filter(data, size)


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


def edges(data):
    border = cv2.borderInterpolate(0, 1, cv2.BORDER_CONSTANT)
    sc = numpy.vectorize(complex)(
        cv2.Sobel(data, cv2.CV_64F, 1, 0, ksize=5, borderType=border),
        cv2.Sobel(data, cv2.CV_64F, 0, 1, ksize=5, borderType=border),
    )
    return numpy.abs(sc), numpy.angle(sc, deg=True)


def find_edges_nd(data):
    magnitude, angle = edges(data)
    return numpy.logical_or(
        numpy.logical_and(
            magnitude > 10 * minimum_filter(magnitude, size=5),
            maximum_filter(magnitude, footprint=Polarizing_Mask.Horizontal)
            == magnitude,
            numpy.vectorize(angle_mask)(angle, 0, 50),
        ),
        numpy.logical_and(
            magnitude > 10 * minimum_filter(magnitude, size=5),
            maximum_filter(magnitude, footprint=Polarizing_Mask.Vertical)
            == magnitude,
            numpy.vectorize(angle_mask)(angle, 90, 50),
        ),
    )


class Polarizing_Mask:
    """Parametric generation of a 2D mask for simple polarization"""

    def _polarized_matrix(
        size: int, angle: float, tolerance: float
    ) -> List[List[bool]]:
        return [
            [
                angle - tolerance
                < numpy.angle(complex(i, j), deg=True)
                < angle + tolerance
                or angle - tolerance
                < numpy.angle(complex(-i, -j), deg=True)
                < angle + tolerance
                or i == 0
                and j == 0
                for i in range(
                    -numpy.floor(size / 2).astype(int),
                    numpy.floor(size / 2).astype(int) + 1,
                )
            ]
            for j in range(
                -numpy.floor(size / 2).astype(int),
                numpy.floor(size / 2).astype(int) + 1,
            )
        ]

    Horizontal = numpy.array(_polarized_matrix(5, 0, 50))
    Diagonal = numpy.array(_polarized_matrix(5, 45, 50))
    Vertical = numpy.array(_polarized_matrix(5, 90, 50))
    IDiagonal = numpy.array(_polarized_matrix(5, -45, 50))


def angle_mask(data: float, angle: float, tolerance: float = 35):
    return (
        True
        if (angle + 180 - tolerance) < (data + 180) < (angle + 180 + tolerance)
        or ((angle + 360) - tolerance)
        < (data + 180)
        < ((angle + 360) + tolerance)
        or ((angle + 360) % 360 - tolerance)
        < (data + 180)
        < ((angle + 360) % 360 + tolerance)
        else False
    )


# =]
