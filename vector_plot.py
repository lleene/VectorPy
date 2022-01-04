"""Image Processing Plotting Procedures"""

from typing import List
from matplotlib import pyplot
import numpy
import cv2
from vector_util import *
import matplotlib
from hilbert import decode

matplotlib.use("tkAgg")


def image_classfied(data, centroids):
    cluster_labels = list(range(max(numpy.array(centroids).shape) + 2))
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


def show_histogram(data):
    pyplot.hist(data, bins=numpy.arange(256), density=True)
    pyplot.yscale("log")
    pyplot.show()


def plot_peaks(data):
    hcount, hloc = numpy.histogram(data, bins=numpy.arange(256), density=True)
    peaks = find_peaks_nd(hcount)
    pyplot.plot(hloc[:-1], hcount)
    pyplot.plot(peaks[0], hcount[peaks[0]], "x")
    pyplot.yscale("log")
    pyplot.show()


def plot_peaks_2d(data):
    hcount, hxloc, hyloc = numpy.histogram2d(
        data[:, :, 0].reshape(data.shape[0] * data.shape[1]),
        data[:, :, 1].reshape(data.shape[0] * data.shape[1]),
        bins=(numpy.arange(256), numpy.arange(256)),
        density=True,
    )
    peakx, peaky_ = find_peaks_nd(hcount)
    pyplot.hist2d(
        data[:, :, 0].reshape(data.shape[0] * data.shape[1]),
        data[:, :, 1].reshape(data.shape[0] * data.shape[1]),
        bins=(numpy.arange(256), numpy.arange(256)),
        density=True,
        norm=matplotlib.colors.LogNorm(),
    )
    pyplot.scatter(peakx, peaky_, marker="o", c="red")
    pyplot.show()


def show_custered_color(data):
    data_c = partion_color(data)
    open_in_window([data, data_c])


def select_channel(data):
    padding = numpy.ones(data[:, :, None].shape) * 128
    return numpy.concatenate((data[:, :, None], padding, padding), axis=2)


def open_in_window(img_list, base_name: str = "sample") -> None:
    for index, image in enumerate(img_list):
        cv2.namedWindow(f"{base_name}_{index}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"{base_name}_{index}", 800, 1600)
        cv2.imshow(f"{base_name}_{index}", image.astype(numpy.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()


def show_classification(file_name):
    image = load_image(file_name)
    centroids = numpy.array(histogram_centroids(image))
    cfd_image = cluster_image(image, centroids)
    show_histogram(find_edges(image, cfd_image))


def show_segmentation(file_name):
    image = load_image(file_name)
    cfd_image = classify_EH_hybrid(image)
    centroids = numpy.unique(cfd_image)
    regions = segment(image, cfd_image)
    open_in_window(
        [image, image_classfied(cfd_image, centroids), regions % 256]
    )


# =]
