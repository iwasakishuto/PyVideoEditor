from PIL import Image
from veditor.utils import draw_cross, SampleData, cv2plot, pil2arr
base = Image.open(SampleData().IMAGE_PATH)
img_square = draw_cross(img=base, size=200, width=10)
img_rect = draw_cross(img=base, size=(100,200), width=10, outline=(0,255,0))
fig, axes = plt.subplots(ncols=3, figsize=(18, 4))
for ax, img in zip(axes, [base, img_square, img_rect]):
    ax = cv2plot(pil2arr(img), ax=ax)
fig.show()
