"""Image Processing Plotting Procedures"""


from matplotlib import pyplot
import numpy
import cv2
from vector_util import find_peaks_nd
import matplotlib

matplotlib.use("tkAgg")


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


def open_in_window(img_list, base_name: str = "sample") -> None:
    for index, image in enumerate(img_list):
        cv2.namedWindow(f"{base_name}_{index}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"{base_name}_{index}", 800, 1600)
        cv2.imshow(f"{base_name}_{index}", image.astype(numpy.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()


# =]
