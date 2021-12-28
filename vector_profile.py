#!/usr/bin/env python3
"""Profile scripts"""

import cProfile
import pstats
import io
from vector_util import *


pr = cProfile.Profile()
pr.enable()

file_name = "cl-sample2.jpg"
image = load_image(file_name)
centroids = numpy.array(histogram_centroids(image))
cfd_image = cluster_image(image, centroids)
regions = segment(image, cfd_image)

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats()
print(s.getvalue())
