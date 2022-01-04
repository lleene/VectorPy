from contour_util import *
from vector_plot import *


from descartes import PolygonPatch



file_name = "cl-sample4.jpg"
image = load_image(file_name)
cfd_image = classify_eh_hybrid(image)
centroids = numpy.unique(cfd_image)
regions = segment(image, cfd_image)
polygons = seed_polygons(region_outlines(regions))


points = numpy.array(numpy.where(regions == 14)).T
points = numpy.array(numpy.where(region_outlines(regions) == 14)).T
fig, ax = pyplot.subplots()
ax.scatter(*zip(*points))
ax.add_patch(PolygonPatch(polygons[14]))
pyplot.show()
