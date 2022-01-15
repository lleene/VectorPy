import numpy
from functools import cmp_to_key
from typing import (
    TypeVar,
    Generic,
    Tuple,
    Union,
    Optional,
    List,
    Dict,
    Tuple,
    NamedTuple,
)
from shapely.geometry import Polygon, MultiPoint, Point
from scipy.ndimage import distance_transform_edt
from vector_util import matching_classes
from tqdm import tqdm


class Color3F(NamedTuple):
    Red: float
    Blue: float
    Green: float


class PolySeed(NamedTuple):
    id: int
    poly: Polygon
    color: Color3F


def region_outlines(regions):
    offsets = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    return numpy.where(
        numpy.any(numpy.logical_not(matching_classes(regions, offsets)), axis=2),
        regions,
        0,
    )


def seed_polygons(regions: numpy.array, image: numpy.array) -> List[PolySeed]:
    outlines = region_outlines(regions)
    keys = list(numpy.unique(outlines))
    collection = {
        region_id: numpy.array(numpy.where(outlines == region_id)).T
        for region_id in keys
        if region_id > 0
    }
    return [
        PolySeed(
            region_id,
            Polygon(estimate_contour(collection[region_id], 6)),
            Color3F(*numpy.mean(image[numpy.where(regions == region_id)], axis=(0))),
        )
        for region_id in tqdm(keys, desc="Tracing")
        if region_id > 0
    ]


def resolve_hierarchy(convex_polygons: List[PolySeed]) -> List[PolySeed]:
    convex_polygons.sort(key=cmp_to_key(polyseed_comparison))
    return convex_polygons


def polyseed_comparison(seeda: PolySeed, seedb: PolySeed) -> int:
    if seeda.poly.is_valid and seeda.poly.covered_by(seedb.poly):
        return 1
    elif seedb.poly.is_valid and seedb.poly.covered_by(seeda.poly):
        return -1
    else:
        return 0


def closest_candidate(points, target, delta, tolerance: float = 1.5):
    """
    Finds the closest point near the target and on the line orthogonal to delta with positive curl.
    """
    unit_delta = delta / numpy.linalg.norm(delta)
    intersect = points[
        numpy.where(
            numpy.logical_and(
                numpy.cross(points - target, unit_delta) > -tolerance,
                abs(numpy.dot(points, unit_delta) - numpy.dot(target, unit_delta))
                < tolerance,
            )
        )
    ]
    return (
        intersect[numpy.argmin(numpy.sum((intersect - target) ** 2, axis=1))]
        if intersect.size
        else numpy.zeros((0, 2))
    )


def find_linear_intersections(points, current_point, delta, count):
    """
    Return a list of interpolated points including starting and ending locations

    These are derived from delta and count and intersect the collection of points
    orthogonally to the vector delta
    """
    return numpy.vstack(
        [current_point] + [
            closest_candidate(points, current_point + delta * elem, delta)
            for elem in range(1, count)
        ] + [(current_point + delta * count)]
    )


def fix_contour_gaps(candidates, points, delta, size, iterations: int = 5):
    """
    fill large gaps in the contour proposed by the candidates

    additional interpolation points are derived by intersecting our points that are again orthogonal to new delta
    that represents a large gap in our proposed contour.
    """
    diff_candidates = numpy.diff(candidates, axis=0)
    candidate_gaps = numpy.where(
        numpy.sum(diff_candidates[:, :] ** 2, axis=1) > 4 * size ** 2
    )[0]
    if candidate_gaps.size <= 0:
        return candidates
    unit_delta = delta / numpy.linalg.norm(delta)
    for gap in candidate_gaps[::-1]:
        current_point = candidates[gap]
        next_point = candidates[gap+1] - numpy.dot(diff_candidates[gap], unit_delta) * unit_delta
        new_count, new_delta = compute_delta(current_point, next_point, size)
        if new_count <= 1:
            continue
        new_candidates = find_linear_intersections(
            points, current_point, new_delta, new_count
        )
        if new_candidates.shape[0] >= 3:
            candidates = numpy.insert(
                candidates,
                gap + 1,
                fix_contour_gaps(
                    new_candidates, points, new_delta, size, iterations - 1
                )[1:-1] if iterations > 1 else new_candidates[1:-1],
                axis=0,
            )
    return candidates


def compute_delta(current_point, next_point, size):
    delta = next_point - current_point
    count = numpy.floor(numpy.linalg.norm(delta) / size).astype(int)
    delta = delta / count if count else delta
    return count, delta


def estimate_contour(
    points: numpy.array,
    size: int,
) -> numpy.array:
    poly_obj = MultiPoint(points).convex_hull
    hull = (
        numpy.array(poly_obj.exterior.xy).T
        if type(poly_obj) == Polygon
        else numpy.array([])
    )
    for index in range(hull.shape[0], 0, -1):
        current_point = hull[index - 1]
        next_point = hull[index % hull.shape[0]]
        count, delta = compute_delta(current_point, next_point, size)
        if count <= 1:
            continue
        candidates = find_linear_intersections(points, current_point, delta, count)
        if candidates.shape[0] >= 3:
            hull = numpy.insert(
                hull,
                index,
                # candidates[1:-1],
                fix_contour_gaps(candidates, points, delta, size)[1:-1, :],
                axis=0,
            )
    return hull


def offset_matrix(size: int = 2) -> numpy.array:
    offsets = (
        numpy.indices((2 * size + 1, 2 * size + 1))
        - numpy.array([size, size])[:, None, None]
    )
    return [
        tuple(elem)
        for elem in offsets.T.reshape((2 * size + 1) * (2 * size + 1), 2)
        if tuple(elem) != (0, 0)
    ]


# eof
