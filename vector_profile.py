#!/usr/bin/env python3
"""Profile scripts"""

import cProfile
import pstats
import io
from vector_util import *
import time


def profile_segment(file_name):
    pr = cProfile.Profile()
    pr.enable()
    image = load_image(file_name)
    centroids = numpy.array(histogram_centroids(image))
    cfd_image = cluster_image(image, centroids)
    regions = segment(image, cfd_image)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()
    print(s.getvalue())


def compare_segment(file_name):
    image = load_image(file_name)
    centroids = numpy.array(histogram_centroids(image))
    cfd_image = cluster_image(image, centroids)
    print(f"Starting comparison ...")
    time1 = time.time()
    regions1 = segment(image, cfd_image)
    time2 = time.time()
    regions2 = segment2(image, cfd_image)
    time3 = time.time()
    print(
        f"Segment1 took {time2-time1:.2f}s and Segment2 took {time3-time2:.2f}s"
    )
    _, counts = numpy.unique(regions1 != regions2, return_counts=True)
    print(f"Found {counts} differences")


file_name = "cl-sample4.jpg"
compare_segment(file_name)
