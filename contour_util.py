from typing import List, Dict, Tuple, NamedTuple
import numpy
from functools import cmp_to_key
from shapely.geometry import Polygon, MultiPoint, Point
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
    # (1, -1), (1, 1), (-1, 1), (-1, -1)]
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
        if region_id > 1
    }
    return [
        PolySeed(
            region_id,
            Polygon(estimate_contour(collection[region_id], 3)),
            Color3F(*numpy.mean(image[numpy.where(regions == region_id)], axis=(0))),
        )
        for region_id in tqdm(keys, desc="Tracing")
        if region_id > 1
    ]


def resolve_hierarchy(convex_polygons: List[PolySeed]) -> List[PolySeed]:
    convex_polygons.sort(key=cmp_to_key(polyseed_comparison))
    return convex_polygons


def polyseed_comparison(seeda: PolySeed, seedb: PolySeed) -> int:
    if seeda.poly.covered_by(seedb.poly):
        return 1
    elif seedb.poly.covered_by(seeda.poly):
        return -1
    else:
        return 0


# rotate_clockwise = numpy.array([[0,-1],[1,0]])

def closest_candidate(points, target, delta, tolerance: float = 3):
    intersect = points[
        numpy.where(
            abs(numpy.dot(points, delta) - numpy.dot(target, delta)) < tolerance
        )
    ]
    return intersect[numpy.argmin(numpy.sum((intersect - target) ** 2, axis=1))] if intersect.size else numpy.zeros((0,2))

def find_linear_intersections(points,current_point, delta, count):
    return numpy.vstack([
        closest_candidate(points, current_point + delta * elem, delta)
        for elem in range(1, count)
    ])

def fix_contour_gaps(candidates, points, delta, size):
    # TODO consider recursive correction method
    diff_candidates = numpy.diff(candidates, axis=0)
    candidate_gaps = numpy.where(numpy.sum(diff_candidates ** 2, axis=1) > 8 * size ** 2)[0]
    for gap in candidate_gaps[::-1]:
        current_point = candidates[gap]
        next_point = candidates[gap+1] - delta
        new_count, new_delta = compute_delta(current_point, next_point, size)
        new_candidates = find_linear_intersections(points,current_point,new_delta,new_count)
        if new_candidates.size >= 1:
            candidates = numpy.insert(candidates, gap + 1, new_candidates, axis=0)
    return candidates

def compute_delta(current_point, next_point, size):
    delta = next_point - current_point
    count = numpy.floor(numpy.linalg.norm(delta) / size).astype(int)
    delta = delta / count if count else delta
    return count, delta

def estimate_contour(points: numpy.array, size: int) -> numpy.array:
    poly_obj = MultiPoint(points).convex_hull
    hull = (
        numpy.array(poly_obj.exterior.xy).T
        if type(poly_obj) == Polygon
        else numpy.array([])
    )
    index = 0
    while index < hull.shape[0]:
        current_point = hull[index]
        next_point = hull[(index + 1) % hull.shape[0]]
        count, delta = compute_delta(current_point,next_point,size)
        if count <= 1:
            index = index + 1
            continue
        candidates = find_linear_intersections(points,current_point, delta, count)
        if candidates.size >= 1:
            hull = numpy.insert(hull, index + 1, fix_contour_gaps(candidates, points, delta, size), axis=0)
        index = index + len(candidates) + 1
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


# =]
