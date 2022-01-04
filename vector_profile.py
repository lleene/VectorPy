#!/usr/bin/env python3
"""Profile scripts"""

import glob
import cProfile
import pstats
import io
from vector_util import *


def profile_segment(file_name):
    pr = cProfile.Profile()
    pr.enable()
    image = load_image(file_name)
    cfd_image = classify_eh_hybrid(image)
    _ = segment(image, cfd_image)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()
    print(s.getvalue())


for file_name in glob.glob("./*.jpg") + glob.glob("./*.png"):
    image = load_image(file_name)
    cfd_image = classify_eh_hybrid(image)
    regions = segment(image, cfd_image)
    cv2.imwrite(f"./temp/{file_name[2:-4]}.dn2.png", regions % 256)
