#!/usr/bin/env python3
"""Color mapping testing scripts"""

from color_util import *
from vector_plot import *


def test_case1(angle: float = None):
    gradient, image_1 = generate_png(
        gradient=LinearGradient(color_angle=angle)
    )
    print(
        f"Testing with {numpy.linalg.norm(gradient.gradient_color):.2f}Δ"
        + f" at {gradient.gradient_angle:.2f} rads"
    )
    fit = LinearGradient(*LinearGradient.fit(image_1, scale=1 / 255))
    gradient_2, image_2 = generate_png(gradient=fit)
    print(
        f"{100 * numpy.linalg.norm(image_1 - image_2) / image_1.size:.2f}% error"
    )
    return image_1, image_2


def test_case2(center=None):
    gradient, image_1 = generate_png(gradient=RadialGradient(center))
    print(
        f"Testing with {numpy.linalg.norm(gradient.gradient_color):.2f}Δ"
        + f" at {gradient.gradient_center}"
    )
    fit = RadialGradient(*RadialGradient.fit(image_1, scale=1 / 255))
    gradient_2, image_2 = generate_png(gradient=fit)
    print(
        f"{100 * numpy.linalg.norm(image_1 - image_2) / image_1.size:.2f}% error"
    )
    return image_1, image_2


# test_case1()
# open_in_window(list(test_case1()))

# test_case2(numpy.array([0.5, 0.5]))
open_in_window(list(test_case2()))

# =]
