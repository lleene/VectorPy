from typing import List, Dict, Tuple, NamedTuple
import numpy
from functools import cmp_to_key
from shapely.geometry import Polygon, MultiPoint
from vector_util import matching_classes


class Color_3F(NamedTuple):
    Red: float
    Blue: float
    Green: float


class PolySeed(NamedTuple):
    id: int
    poly: Polygon
    color: Color_3F


def region_outlines(regions):
    return numpy.where(
        numpy.any(numpy.logical_not(matching_classes(regions)), axis=2), regions, 0
    )


def seed_polygons(regions: numpy.array, image: numpy.array) -> List[PolySeed]:
    outlines = region_outlines(regions)
    keys = numpy.unique(outlines)
    collection = {
        region_id: numpy.array(numpy.where(outlines == region_id)).T
        for region_id in keys
        if region_id > 1
    }
    return [
        PolySeed(
            region_id,
            MultiPoint(collection[region_id]).convex_hull,
            Color_3F(*numpy.mean(image[numpy.where(regions == region_id)], axis=(0))),
        )
        for region_id in collection
    ]


def resolve_hierarchy(convex_polygons: List[PolySeed]) -> List[PolySeed]:
    convex_polygons.sort(key=cmp_to_key(polyseed_comparison))
    return convex_polygons


def concave_hull(points: numpy.array) -> Polygon:
    # TODO fine tune polygon points if needed
    pass


def polyseed_comparison(seeda: PolySeed, seedb: PolySeed) -> int:
    if seeda.poly.covered_by(seedb.poly):
        return 1
    elif seedb.poly.covered_by(seeda.poly):
        return -1
    else:
        return 0


# =]
