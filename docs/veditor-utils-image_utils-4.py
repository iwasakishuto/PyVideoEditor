import cv2
import matplotlib.pyplot as plt
from veditor.utils import cv2plot, SampleData, nega_conversion
frame = cv2.imread(SampleData().IMAGE_PATH)
fig, axes = plt.subplots(ncols=2, figsize=(6*2, 4))
ax = cv2plot(frame, ax=axes[0])
ax.set_title("Original")
ax = cv2plot(nega_conversion(frame=frame), ax=axes[1])
ax.set_title("Nega-posi Conversion")
fig.show()
