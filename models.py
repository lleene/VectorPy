from typing import TypeVar, Generic, Tuple, Union, Optional, List, Dict, Tuple, NamedTuple
from shapely.geometry import Polygon, MultiPoint, Point
import numpy as np

Shape = TypeVar("Shape")
DType = TypeVar("DType")

class Array(np.ndarray, Generic[Shape, DType]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """
    pass

