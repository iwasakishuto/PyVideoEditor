import cv2
import matplotlib.pyplot as plt
from veditor.utils import (
    SUPPORTED_CONVERSION_METHODS,
    cv2plot,
    image_conversion,
    SampleData,
)
frame = cv2.imread(SampleData().IMAGE_PATH)
num_methods = len(SUPPORTED_CONVERSION_METHODS)
ncols = 3; nrows = num_methods//ncols
fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6 * ncols, 4 * nrows))
for i,method in enumerate(SUPPORTED_CONVERSION_METHODS):
    ax = cv2plot(image_conversion(frame, method=method), ax=axes[i%2][i//2])
    ax.set_title(method)
fig.show()
