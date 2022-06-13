from typing import List, Tuple, Union
import numpy
import cv2


def derivative_features(image, size: int = 5):
    # TODO this may be throwing away radial information for negative deltas
    scale = 1/numpy.sum(numpy.abs(numpy.outer(*cv2.getDerivKernels(1, 0, size))))
    components = numpy.vectorize(complex)(
                numpy.linalg.norm(cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=size, scale=scale, borderType=cv2.BORDER_REFLECT),axis=2),
                numpy.linalg.norm(cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=size, scale=scale, borderType=cv2.BORDER_REFLECT),axis=2),
        )
    return numpy.abs(components).astype(int), numpy.angle(components, deg=True).astype(int)


def load_image(file_name: str, denoise_factor: int = None):
    img = cv2.imread(file_name)
    return cv2.cvtColor(
        cv2.fastNlMeansDenoisingColored(img, h=denoise_factor, hColor=denoise_factor)
        if denoise_factor
        else img,
        cv2.COLOR_BGR2RGB,
    )


def shift_image(image: numpy.array, shift: int, cval: Union[float, int]):
    result = numpy.empty_like(image)
    if shift == (1, 0):
        result[:1, :] = cval
        result[1:, :] = image[:-1, :]
    elif shift == (0, 1):
        result[:, :1] = cval
        result[:, 1:] = image[:, :-1]
    elif shift == (-1, 0):
        result[-1:, :] = cval
        result[:-1, :] = image[1:, :]
    elif shift == (0, -1):
        result[:, -1:] = cval
        result[:, :-1] = image[:, 1:]
    elif shift == (-1, -1):
        result[-1:, :] = cval
        result[:, -1:] = cval
        result[:-1, :-1] = image[1:, 1:]
    elif shift == (1, 1):
        result[:1, :] = cval
        result[:, :1] = cval
        result[1:, 1:] = image[:-1, :-1]
    elif shift == (1, -1):
        result[:1, :] = cval
        result[:, -1:] = cval
        result[1:, :-1] = image[:-1, 1:]
    elif shift == (-1, 1):
        result[-1:, :] = cval
        result[:, 1] = cval
        result[:-1, 1:] = image[1:, :-1]
    else:
        raise ValueError(f"Unspecified shift value: {shift}")
    return result