import cv2
import matplotlib.pyplot as plt
from veditor.utils import cv2plot, SampleData, min_max_normalization
frame = cv2.imread(SampleData().IMAGE_PATH)
fig, axes = plt.subplots(ncols=2, figsize=(6*2, 4))
ax = cv2plot(frame, ax=axes[0])
ax.set_title("Original")
ax = cv2plot(min_max_normalization(frame=frame), ax=axes[1])
ax.set_title("min-max Normalization")
fig.show()
