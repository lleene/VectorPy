#!/usr/bin/env python3
"""Color mapping testing scripts"""

import cairo
import numpy
from typing import Tuple

# from vector_plot import open_in_window


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
    def linear_features(cls, image):
        """Generates feature space for linear mapping with linear gradient."""
        X = numpy.indices(image[:, :, 0].shape).T.reshape(
            image.shape[0] * image.shape[1], 2
        )
        Y = image.reshape(image.shape[0] * image.shape[1], *image.shape[2:])
        return (
            numpy.concatenate([X, numpy.ones(X.shape[0])[:, None]], axis=1),
            Y,
        )

    @classmethod
    def fit(cls, image, scale: float = 1.0):
        cfit, rfit = numpy.linalg.lstsq(
            *cls.linear_features(image), rcond=None
        )[:2]
        print(
            f"{100*numpy.linalg.norm(rfit) / image.size :.2f}% LLS residual error."
        )
        assert numpy.linalg.norm(rfit) / image.size < 0.1  # tolerate 10% error
        angle = numpy.arctan(
            numpy.divide(*list(numpy.linalg.norm(cfit[:-1, :], axis=1)))
            * numpy.sign(numpy.dot(cfit[0, :], cfit[1, :]))
        )
        outline = numpy.max(image.shape)
        if angle > 0:
            delta = numpy.array(
                [
                    [0, outline * 1.5 * numpy.sin(angle)],
                    [0, outline * 1.5 * numpy.cos(angle)],
                ]
            )
        else:
            delta = numpy.array(
                [
                    [outline, outline + outline * 1.5 * numpy.sin(angle)],
                    [0, outline * 1.5 * numpy.cos(angle)],
                ]
            )
        return (
            angle,
            (
                numpy.dot(cfit[:2, :].T, delta)
                + numpy.repeat(cfit[-1, None].T, 2, axis=1)
            )[
                :-1, :
            ]  # ignore alpha for now
            * scale,
        )


class Radial_Gradient:
    def __init__(
        self,
        color_center: Tuple[float, float] = None,
        color_step=None,
    ):
        self.gradient_center = (
            numpy.random.uniform(low=0.0, high=1.0, size=(2))
            if not hasattr(color_center, "shape")
            else color_center
        )
        self.gradient_color = (
            numpy.random.uniform(low=0.0, high=1.0, size=(3, 2))
            if not hasattr(color_step, "shape")
            else color_step
        )

    @property
    def cairo_source(self):
        pat = cairo.RadialGradient(
            *list(self.gradient_center), 0.0, *list(self.gradient_center), 1.5
        )
        pat.add_color_stop_rgba(0, *self.gradient_color[::-1, 0], 1)
        pat.add_color_stop_rgba(1, *self.gradient_color[::-1, 1], 1)
        # pat.set_matrix(self.transform)
        return pat

    @classmethod
    def radial_features(cls, image):
        """
        Generates feature space for concentric mapping with linear gradient.

        Generate concentric features fitting the eq.  a0 × x² + a1 × y² + a2 = z²
        F1 = R × sin²(theta), F2 = R × cos²(theta) where R² = x² + y² and θ = arctan(x/y)
        This simplifies to F1 = x² / sqrt(x² + y²), F2 = y² / sqrt(x² + y²)

        The basis is normalized to the unit square and offset such that radial components
        are centered mid point.
        """
        X = numpy.indices(image[:, :, 0].shape).T.reshape(
            image.shape[0] * image.shape[1], 2
        ) - round(numpy.max(image.shape) / 2)
        A = numpy.divide(
            X ** 2,
            numpy.sqrt(numpy.sum(X ** 2, axis=1))[:, None],
            where=numpy.isclose(X, 0.0, atol=1e-3).all(axis=1)[:, None]
            == False,
        )
        Y = image.reshape(image.shape[0] * image.shape[1], *image.shape[2:])
        return (
            numpy.concatenate([X, A, numpy.ones(X.shape[0])[:, None]], axis=1),
            Y,
        )

    @classmethod
    def fit(cls, image, scale: float = 1.0):
        cfit, rfit = numpy.linalg.lstsq(
            *cls.radial_features(image), rcond=None
        )[:2]
        print(
            f"LLS residuals: {numpy.linalg.norm(rfit)} over {image.size} samples"
        )
        assert numpy.linalg.norm(rfit) < image.size
        outline = numpy.max(image.shape)
        delta = numpy.array([[0.0, 1.5 * outline], [0.0, 1.5 * outline]])
        center = (
            outline / 2
            - numpy.sum(cfit[2:4, :-1] / cfit[0:2, :-1], axis=1) / 3
        )
        return (
            center,
            (
                numpy.dot(cfit[2:4, :].T, delta)
                + numpy.repeat(cfit[-1, None].T, 2, axis=1)
            )[
                :-1, :
            ]  # ignore alpha for now
            * scale,
        )


def generate_L1_PNG(gradient, width: int = 100, height: int = 100):
    data = numpy.ndarray(shape=(height, width, 4), dtype=numpy.uint8)
    surface = cairo.ImageSurface.create_for_data(
        data, cairo.FORMAT_ARGB32, width, height
    )
    ctx = cairo.Context(surface)
    ctx.scale(width, height)
    ctx.rectangle(0, 0, 1, 1)
    ctx.set_source(gradient.cairo_source)
    ctx.fill()
    return (
        gradient,
        data,
    )


def test_case1(angle: float = None):
    gradient, image_1 = generate_L1_PNG(
        gradient=Linear_Gradient(color_angle=angle)
    )
    print(
        f"Testing with {numpy.linalg.norm(gradient.gradient_color):.2f}"
        + f" at {gradient.gradient_angle:.2f}"
    )
    angle, step = Linear_Gradient.fit(image_1, scale=1 / 255)
    fit = Linear_Gradient(angle, step)
    gradient_2, image_2 = generate_L1_PNG(gradient=fit)
    return image_1, image_2


def test_case2(angle: float = None):
    gradient, image_1 = generate_L1_PNG(gradient=Radial_Gradient())
    print(
        f"Testing with {numpy.linalg.norm(gradient.gradient_color):.2f}"
        + f" at {gradient.gradient_center}"
    )
    center, step = Radial_Gradient.fit(image_1, scale=1 / 255)
    fit = Radial_Gradient(center, step)
    gradient_2, image_2 = generate_L1_PNG(gradient=fit)
    return image_1, image_2


# test_case1()
test_case2()


# =]
