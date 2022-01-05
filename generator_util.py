from vector_util import *
from contour_util import *
import cairo
from tqdm import tqdm


def generate_data_context(height, width) -> cairo.Context:
    data = numpy.ndarray(shape=(height, width, 4), dtype=numpy.uint8)
    surface = cairo.ImageSurface.create_for_data(
        data, cairo.FORMAT_ARGB32, width, height
    )
    return data, cairo.Context(surface)


def draw_polygons_context(ctx, polygons):
    for poly_id, poly_obj, color in tqdm(
        resolve_hierarchy(polygons), desc="Constructing"
    ):
        if type(poly_obj) != Polygon:
            continue
        points = numpy.array(poly_obj.exterior.xy)
        ctx.move_to(points[1, 0], points[0, 0])
        for index in range(1, points.shape[1] - 1):
            ctx.line_to(points[1, index], points[0, index])
        ctx.close_path()
        ctx.set_source_rgba(color.Red / 256, color.Blue / 256, color.Green / 256, 1)
        ctx.fill()


def dump_to_svg_file(file_name):
    image = load_image(file_name)
    cfd_image = classify_eh_hybrid(image)
    regions = segment(image, cfd_image)
    polygons = seed_polygons(regions, image)
    with cairo.SVGSurface("temp.svg", image.shape[1], image.shape[0]) as surface:
        ctx = cairo.Context(surface)
        draw_polygons_context(ctx, polygons)
        surface.finish()


def dump_to_data(file_name):
    image = load_image(file_name)
    cfd_image = classify_eh_hybrid(image)
    regions = segment(image, cfd_image)
    polygons = seed_polygons(regions, image)
    data = numpy.ndarray(shape=(image.shape[0], image.shape[1], 4), dtype=numpy.uint8)
    surface = cairo.ImageSurface.create_for_data(
        data, cairo.FORMAT_RGB24, image.shape[1], image.shape[0]
    )
    ctx = cairo.Context(surface)
    draw_polygons_context(ctx, polygons)
    surface.finish()
    return data[:, :, :3]
