#!/usr/bin/env python3
"""Testing scripts"""

# from sklearn.cluster import MeanShift, estimate_bandwidth  # SpectralClustering
from matplotlib import pyplot
import numpy
from scipy.signal import find_peaks
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage.morphology import generate_binary_structure
import cv2
import matplotlib
from hilbert import decode, encode

matplotlib.use("tkAgg")


def image_classfied(data, cluster_labels):
    bits = round(numpy.ceil(numpy.log2(len(cluster_labels))))
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


def open_in_window(base_name: str, img_list) -> None:
    for index, image in enumerate(img_list):
        cv2.namedWindow(f"{base_name}_{index}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"{base_name}_{index}", 800, 1600)
        cv2.imshow(f"{base_name}_{index}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def is_monochrome(image) -> bool:
    return image[:, :, 1:].std() < 5.0


def load_image(file_name: str):
    img = cv2.imread(file_name)
    yrc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    border = cv2.borderInterpolate(0, 1, cv2.BORDER_CONSTANT)
    lpc = cv2.Laplacian(yrc, cv2.CV_64F, ksize=3, borderType=border)
    return yrc, lpc


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


def find_peaks_nd(data, size=5):
    return numpy.where(
        numpy.logical_and(
            minimum_filter(data, size=size) > 0.0,
            maximum_filter(data, size=size) == data,
            numpy.divide(data, minimum_filter(data, size=size)) >= 2,
        )
    )


def cluster_color(image, centroids_2d):
    data = image.reshape(image.shape[0] * image.shape[1], 3)
    return numpy.argmin(
        abs(
            data[:, 1]
            - centroids_2d[0][:, numpy.newaxis]
            + data[:, 2]
            - centroids_2d[1][:, numpy.newaxis]
        ),
        axis=0,
    ).reshape(image.shape[0], image.shape[1])


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


def seed_centroids(data):
    if is_monochrome(data):
        hcount, _ = numpy.histogram(
            data[:, :, 0].reshape(data.shape[0] * data.shape[1]),
            bins=numpy.arange(256),
            density=True,
        )
    else:
        hcount, _, _ = numpy.histogram2d(
            data[:, :, 1].reshape(data.shape[0] * data.shape[1]),
            data[:, :, 2].reshape(data.shape[0] * data.shape[1]),
            bins=(numpy.arange(256), numpy.arange(256)),
            density=True,
        )
    return find_peaks_nd(hcount)


image, edges = load_image("cl-sample1.jpg")
# plot_peaks_2d(image)
centroids = numpy.array(seed_centroids(image))
result = cluster_color(image[::4, ::4, :], centroids)
c_image = image_classfied(result, numpy.arange(centroids.shape[1]))
open_in_window("sample", [image, c_image])

# sub = image[0::8, 0::8, 0]
# sub = sub.reshape(sub.shape[0] * sub.shape[1], 1)

# data = numpy.concatenate((image, edges), axis=2)
# xdata = data.reshape(data.shape[0] * data.shape[1], 6)
# bandwidth = estimate_bandwidth(sub, quantile=0.2, n_samples=500)
# ms = MeanShift(bandwidth=bandwidth)

# if is_monochrome(image):  # use lumina component for coarse grouping
#     print("is_monochrome")
# else:  # use color component for coarse grouping
#     print("is_color")


# open_in_window("sample", [image])
# bandwidth = estimate_bandwidth(sub, quantile=0.2, n_samples=500)
# ms = MeanShift(bandwidth=bandwidth)

# if is_monochrome(image):  # use lumina component for coarse grouping
#     print("is_monochrome")
# else:  # use color component for coarse grouping
#     print("is_color")


# open_in_window("sample", [image])
# ms = MeanShift(bandwidth=bandwidth)

# if is_monochrome(image):  # use lumina component for coarse grouping
#     print("is_monochrome")
# else:  # use color component for coarse grouping
#     print("is_color")


# open_in_window("sample", [image])
# if is_monochrome(image):  # use lumina component for coarse grouping
#     print("is_monochrome")
# else:  # use color component for coarse grouping
#     print("is_color")


# open_in_window("sample", [image])
#     print("is_monochrome")
# else:  # use color component for coarse grouping
#     print("is_color")


# open_in_window("sample", [image])

# open_in_window("sample", [image])
#     print("is_monochrome")
# else:  # use color component for coarse grouping
#     print("is_color")


# open_in_window("sample", [image])


# open_in_window("sample", [image])
