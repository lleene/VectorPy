"""Subpixel Line Detection scripts"""

import numpy


def derivative_features(image, size: int = 5):
    # TODO this may be throwing away radial information for negative deltas
    scale = 1/numpy.sum(numpy.abs(numpy.outer(*cv2.getDerivKernels(1, 0, size))))
    components = numpy.vectorize(complex)(
                numpy.linalg.norm(cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=size, scale=scale, borderType=cv2.BORDER_REFLECT),axis=2),
                numpy.linalg.norm(cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=size, scale=scale, borderType=cv2.BORDER_REFLECT),axis=2),
        )
    return numpy.abs(components).astype(int), numpy.angle(components, deg=True).astype(int)

# for all detected edges calculate single-double edge parameters
# find the sub-pixel edge to determine edge-continuity and specific sub-set


