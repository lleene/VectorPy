#!/usr/bin/env python3
"""Testing scripts"""

from vector_util import *
from vector_plot import *

img = load_image("bw-sample1.jpg")
result, centroids = partion_image(img)


def inverse_mask(image_data, image_mask):
    return 0 if not image_mask else image_data


estimate = denoise_class(
    numpy.vectorize(inverse_mask)(
        result,
        cv2.Laplacian(
            result.astype(numpy.float64), cv2.CV_64F, ksize=3
        ).astype(bool),
    )
)

line_estimate = (estimate == 0).astype(int)

cimg = image_classfied(line_estimate, [0, 1])

open_in_window([img, line_estimate])

# sub = image[0::8, 0::8, 0]
# sub = sub.reshape(sub.shape[0] * sub.shape[1], 1)

# data = numpy.concatenate((image, edges), axis=2)
# xdata = data.reshape(data.shape[0] * data.shape[1], 6)
# bandwidth = estimate_bandwidth(sub, quantile=0.2, n_samples=500)
# ms = MeanShift(bandwidth=bandwidth)

# if is_monochrome(image):  # use lumina component for coarse grouping
#     print("is_monochrome")
# else:  # use color component for coarse grouping
#     print("is_color")


# open_in_window("sample", [image])
# bandwidth = estimate_bandwidth(sub, quantile=0.2, n_samples=500)
# ms = MeanShift(bandwidth=bandwidth)

# if is_monochrome(image):  # use lumina component for coarse grouping
#     print("is_monochrome")
# else:  # use color component for coarse grouping
#     print("is_color")


# open_in_window("sample", [image])
# ms = MeanShift(bandwidth=bandwidth)

# if is_monochrome(image):  # use lumina component for coarse grouping
#     print("is_monochrome")
# else:  # use color component for coarse grouping
#     print("is_color")


# open_in_window("sample", [image])
# if is_monochrome(image):  # use lumina component for coarse grouping
#     print("is_monochrome")
# else:  # use color component for coarse grouping
#     print("is_color")


# open_in_window("sample", [image])
#     print("is_monochrome")
# else:  # use color component for coarse grouping
#     print("is_color")


# open_in_window("sample", [image])

# open_in_window("sample", [image])
#     print("is_monochrome")
# else:  # use color component for coarse grouping
#     print("is_color")
# open_in_window("sample", [image])
#     print("is_monochrome")
# else:  # use color component for coarse grouping
#     print("is_color")


# open_in_window("sample", [image])

# open_in_window("sample", [image])
#     print("is_monochrome")
# else:  # use color component for coarse grouping
#     print("is_color")
# else:  # use color component for coarse grouping
#     print("is_color")
# open_in_window("sample", [image])
#     print("is_monochrome")
# else:  # use color component for coarse grouping
#     print("is_color")
# else:  # use color component for coarse grouping
#     print("is_color")
#     print("is_color")
#     print("is_color")
# else:  # use color component for coarse grouping
#     print("is_color")
# else:  # use color component for coarse grouping
#     print("is_color")
#     print("is_color")
#     print("is_color")
