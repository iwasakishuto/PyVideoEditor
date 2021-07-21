# coding: utf-8

import os
from numbers import Number
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from PIL import Image

from ..utils._colorings import toBLUE, toGREEN
from ..utils.image_utils import alpha_composite, arr2pil, cv2plot, pil2arr
from .base import BaseElement, FixedElement


class ImageElement(FixedElement):
    def __init__(
        self,
        x: Union[str, npt.NDArray[np.uint8]],
        pos_frames: Tuple[int, Optional[int]] = (0, None),
        margin: Union[Number, List[Number]] = 0,
        dsize: Optional[Tuple[int, int]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        top: Optional[Union[BaseElement, int]] = None,
        right: Optional[Union[BaseElement, int]] = None,
        left: Optional[Union[BaseElement, int]] = None,
        bottom: Optional[Union[BaseElement, int]] = None,
    ):
        """Image Elements.

        Args:
            x (Union[str, npt.NDArray[np.uint8]])                : An image like array or the path to the image file.
            pos_frames (Tuple[int, Optional[int]], optional)     : Start and end positions. Defaults to ``(0, None)``.
            margin (Optional[Union[int, List[int]]], optional)   : Margin. Defaults to ``None``.
            dsize (Optional[Tuple[int, int]], optional)          : Desired size for an image. Defaults to ``None``.
            width (Optional[int], optional)                      : The element width. Defaults to ``None``.
            height (Optional[int], optional)                     : The element height. Defaults to ``None``.
            top (Optional[Union[BaseElement, int]], optional)    : Reference element or absolute value at the top. Defaults to ``None``.
            right (Optional[Union[BaseElement, int]], optional)  : Reference element or absolute value at the right. Defaults to ``None``.
            left (Optional[Union[BaseElement, int]], optional)   : Reference element or absolute value at the left. Defaults to ``None``.
            bottom (Optional[Union[BaseElement, int]], optional) : Reference element or absolute value at the bottom. Defaults to ``None``.

        .. plot::
            :class: popup-img

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from veditor.elements import ImageElement
            >>> from veditor.utils import SampleData, cv2plot
            >>> W,H = (1280, 720)
            >>> fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(6*2, 4))
            >>> bg = np.full(shape=(H,W,3), fill_value=(0,255,0), dtype=np.uint8)
            >>> cv2plot(bg, ax=ax1)
            >>> element = ImageElement(x=SampleData().IMAGE_PATH, left=0, right=W, top=0, bottom=H)
            >>> cv2plot(element.edit(bg, pos=0), ax=ax2)
            >>> fig.show()
        """
        super().__init__(
            pos_frames=pos_frames,
            margin=margin,
            width=width,
            height=height,
            top=top,
            right=right,
            left=left,
            bottom=bottom,
            **dict(dsize=dsize, x=x),
        )

    def set_size(
        self,
        x: Union[str, npt.NDArray[np.uint8]],
        dsize: Optional[Tuple[int, int]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """Set

        Args:
            x (Union[str, npt.NDArray[np.uint8]])       : [description].
            dsize (Optional[Tuple[int, int]], optional) : [description]. Defaults to ``None``.
            width (Optional[int], optional)             : [description]. Defaults to ``None``.
            height (Optional[int], optional)            : [description]. Defaults to ``None``.
        """
        self.set_image_attributes(x=x)
        self.resize(dsize=dsize, width=width, height=height)

    def resize(
        self,
        dsize: Optional[Tuple[int, int]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """Resize the both ``image_pil`` and ``image_arr`` attributes.

        If only ``width`` or ``height`` is given, resize while preserving the aspect ratio.

        Args:
            dsize (Optional[Tuple[int,int]], optional) : Desired image size. Defaults to ``None``.
            width (Optional[int], optional)            : Desired image width. Defaults to ``None``.
            height (Optional[int], optional)           : Desired image height. Defaults to ``None``.
        """
        if dsize is None:
            if width is None:
                if height is None:
                    self.logger.warn(
                        f"If you want to resize the image, please specify at least one of {toBLUE('dsize')}, {toBLUE('width')}, {toBLUE('height')}."
                    )
                    width = self.width
                    height = self.height
                else:
                    width = int(self.width * (height / self.height))
            elif height is None:
                height = int(self.height * (width / self.width))
        else:
            width, height = dsize
        self.pil = self.pil.resize((width, height))
        self.arr = cv2.resize(self.arr, dsize=(width, height))
        self.set_attribute(name="width", value=width)
        self.set_attribute(name="height", value=height)

    def edit(self, frame: npt.NDArray[np.uint8], pos: int) -> npt.NDArray[np.uint8]:
        """Edit a ``pos``-th frame in the video ``vide_path``.

        Args:
            frame (npt.NDArray[np.uint8]) : The current frame (BGR image) in the video.
            pos (int)                     : The current position in the video.

        Returns:
            npt.NDArray[np.uint8]: An editied frame.
        """
        if self.arr.ndim == 3:
            frame[self.top : self.bottom, self.left : self.right, :] = self.arr[
                :, :, :3
            ]
        else:
            img = alpha_composite(
                bg=arr2pil(frame), paste=self.pil, box=(self.left, self.top)
            )
            frame = pil2arr(img)
        return frame

    def set_image_attributes(
        self,
        x: Union[str, npt.NDArray[np.uint8]],
    ) -> None:
        """Set attributes for an image.

        Args:
            x (Union[str, npt.NDArray[np.uint8]])                    : An image like array or the path to the image file.
            margin (Optional[Union[Number, List[Number]]], optional) : Margin. Defaults to ``None``.
            margin_default (Number, optional)                        : Default value for margin. Defaults to ``0``.

        Raises:
            FileNotFoundError: When file is not found.

        Examples:
            >>> from veditor.utils import SampleData
            >>> from veditor.elements import ImageElement
            >>> element = ImageElement(x=SampleData().IMAGE_PATH)
            >>> hasattr(element, "arr") and hasattr(element, "pil")
            True
        """
        if isinstance(x, str):
            if not os.path.exists(x):
                raise FileNotFoundError(f"{toBLUE(x)} is not found.")
            img_pil = Image.open(x)
            x = cv2.imread(x)
        else:
            img_pil = arr2pil(x)
        width, height = img_pil.size
        self.set_attribute(name="pil", value=img_pil, msg=f"size={img_pil.size}")
        self.set_attribute(name="arr", value=x, msg=f"shape={x.shape}")
        self.set_attribute(name="width", value=width)
        self.set_attribute(name="height", value=height)

    def show_image_arr(self, ax: Optional[Axes] = None) -> Axes:
        """Show ``image_arr`` using :func:`cv2plot <veditor.utils.image_utils.cv2plot>`

        Args:
            ax (Optional[Axes], optional) : An ``Axes`` instance. Defaults to ``None``.

        Returns:
            Axes: An ``Axes`` instance with ``frame`` drawn.
        """
        ax = cv2plot(frame=self.arr, ax=ax, isBGR=True)
        return ax
