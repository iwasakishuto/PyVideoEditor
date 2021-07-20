import cv2
import matplotlib.pyplot as plt
from veditor.utils import (
    SUPPORTED_CONVERSION_METHODS,
    cv2plot,
    image_conversion,
)
frame = cv2.imread()
num_methods = len(SUPPORTED_CONVERSION_METHODS)
fig, axes = plt.subplots(ncols=num_methods, nrows=1, figsize=(6 * num_methods, 4))
for ax, method in zip(axes, SUPPORTED_CONVERSION_METHODS):
    ax = cv2plot(image_conversion(frame, method=method), ax=ax)
    ax.set_title(method)
fig.show()
