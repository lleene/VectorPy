#!/usr/bin/env python3
"""Testing scripts"""

import cairo
import numpy
from sklearn.linear_model import LinearRegression
from vector_plot import *


class Linear_Gradient:
    def __init__(self, color_angle: float = None, color_step=None):
        self.gradient_angle = (
            numpy.random.uniform(low=-numpy.pi / 2, high=numpy.pi / 2)
            if not isinstance(color_angle, (float, int))
            else color_angle
        )
        self.gradient_color = (
            numpy.random.uniform(low=0.0, high=1.0, size=(3, 2))
            if not hasattr(color_step, "shape")
            else color_step
        )

    def __bool__(self):
        return (
            True
            if self.gradient_angle and hasattr(self.gradient_color, "shape")
            else False
        )

    @property
    def normal_vector(self):
        delta = self.gradient_color[:, 1] - self.gradient_color[:, 0]
        return delta / numpy.linalg.norm(delta)

    @property
    def offset(self):
        return (
            self.gradient_color[:, 0]
            if self.gradient_angle > 0
            else self.gradient_color[:, 0] - self.normal_vector[0],
        )

    @property
    def cairo_source(self):
        delta_x = numpy.sin(self.gradient_angle) * 1.5
        delta_y = numpy.cos(self.gradient_angle) * 1.5
        if delta_x > 0:
            pat = cairo.LinearGradient(0, 0, delta_x, delta_y)
        else:
            pat = cairo.LinearGradient(1, 0, 1 + delta_x, delta_y)
        pat.add_color_stop_rgba(0, *self.gradient_color[::-1, 0], 1)
        pat.add_color_stop_rgba(1, *self.gradient_color[::-1, 1], 1)
        return pat

    @classmethod
    def fit(self, pixel_index, pixel_data):
        fit = LinearRegression().fit(pixel_index, pixel_data)
        angle = numpy.arctan(
            numpy.divide(*list(numpy.linalg.norm(fit.coef_, axis=0)))
        ) * numpy.sign(numpy.dot(fit.coef_[:-1, 0], fit.coef_[:-1, 1]))
        if angle > 0:
            delta = numpy.array(
                [
                    [0, 1.5 * numpy.sin(angle)],
                    [0, 1.5 * numpy.cos(angle)],
                ]
            )
        else:
            delta = numpy.array(
                [
                    [1, 1 + 1.5 * numpy.sin(angle)],
                    [0, 1.5 * numpy.cos(angle)],
                ]
            )
        return (
            angle,
            (
                numpy.dot(fit.coef_, delta)
                + numpy.repeat(fit.intercept_[:, None], 2, axis=1)
            )[:-1, :]
            / 255,
        )


def generate_L1_PNG(
    width: int = 100, height: int = 100, gradient: Linear_Gradient = None
):
    data = numpy.ndarray(shape=(height, width, 4), dtype=numpy.uint8)
    surface = cairo.ImageSurface.create_for_data(
        data, cairo.FORMAT_ARGB32, width, height
    )
    ctx = cairo.Context(surface)
    ctx.scale(width, height)
    ctx.rectangle(0, 0, 1, 1)
    ctx.set_source(
        gradient.cairo_source if gradient else Linear_Gradient().cairo_source
    )
    ctx.fill()
    return (
        gradient,
        data,
    )


def get_xy(image):
    X = numpy.indices(image[:, :, 0].shape).transpose().reshape(
        image.shape[0] * image.shape[1], 2
    ) / numpy.max(
        image.shape
    )  # test case is fillin the unit square
    Y = image.reshape(image.shape[0] * image.shape[1], *image.shape[2:])
    return X, Y


def test_case(angle: float = None):
    gradient, image = generate_L1_PNG(
        gradient=Linear_Gradient(color_angle=angle)
    )
    angle, step = Linear_Gradient.fit(*get_xy(image))
    fit = Linear_Gradient(angle, step)
    gradient_2, image_2 = generate_L1_PNG(gradient=fit)
    open_in_window([image, image_2])


# =]
