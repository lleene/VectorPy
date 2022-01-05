from typing import List, Dict, Tuple, NamedTuple
import numpy
from functools import cmp_to_key
from shapely.geometry import Polygon, MultiPoint
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
        delta = next_point - current_point
        count = numpy.floor(numpy.linalg.norm(delta) / size).astype(int)
        if count <= 1:
            index = index + 1
            continue
        delta = delta / count
        targets = (
            numpy.repeat(numpy.arange(1, count)[:, None], (2), axis=1) * delta
            + current_point
        )
        candidates = [
            numpy.argmin(numpy.sum((points[:, :] - elem) ** 2, axis=1))
            for elem in targets
        ]
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) >= 1:
            hull = numpy.insert(hull, index+1, points[candidates], axis=0)
        index = index + len(candidates) + 1
    return hull


def offset_matrix(size: int = 2) -> numpy.array:
    # Dict[Tuple[int,int],int]:
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
