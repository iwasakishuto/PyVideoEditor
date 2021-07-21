import cv2
import matplotlib.pyplot as plt
from veditor.utils import cv2plot, SampleData, apply_heatmap
frame = cv2.imread(SampleData().IMAGE_PATH)
colormaps = ["Pastel1", "Set1", "tab10", "hsv", "bwr", "Reds"]
num_methods = len(colormaps)
ncols = 3; nrows = num_methods//ncols
fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6 * ncols, 4 * nrows))
for i,cmap in enumerate(colormaps):
    ax = cv2plot(apply_heatmap(frame, cmap=cmap), ax=axes[i%2][i//2])
    ax.set_title(cmap)
fig.show()
