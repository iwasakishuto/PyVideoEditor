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
from ..utils.generic_utils import assign_trbl
from ..utils.image_utils import alpha_composite, arr2pil, cv2plot, pil2arr
from .base import BaseElement


class ImageElement(BaseElement):
    def __init__(
        self,
        path: str,
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
        super().__init__(pos_frames=pos_frames)
        self.set_image_attributes(path, margin=margin)
        self.resize(dsize=dsize, width=width, height=height)
        self.set_locations(top=top, right=right, left=left, bottom=bottom)

    def edit(self, frame: npt.NDArray[np.uint8], pos: int) -> npt.NDArray[np.uint8]:
        """Edit a ``pos``-th frame in the video ``vide_path``.

        Args:
            frame (npt.NDArray[np.uint8]) : The current frame (BGR image) in the video.
            pos (int)                     : The current position in the video.

        Returns:
            npt.NDArray[np.uint8]: An editied frame.
        """
        if self.image_arr.ndim == 3:
            frame[self.top : self.bottom, self.left : self.right, :] = self.image_arr[
                :, :, :3
            ]
        else:
            img = alpha_composite(
                bg=arr2pil(frame), paste=self.image_pil, box=(self.left, self.top)
            )
            frame = pil2arr(img)
        return frame

    def set_image_attributes(
        self,
        path,
        margin: Optional[Union[Number, List[Number]]] = None,
        margin_default: Number = 0,
    ) -> None:
        """Set attributes for an image.

        Raises:
            FileNotFoundError: When file is not found.

        Examples:
            >>> from veditor.utils import
            >>> from veditor. import RotatingRectangleEditor
            >>> editor = RotatingRectangleEditor()
            >>> editor.set_image_attributes(hoge_image=ROTATING_SQUARE_IMAGE_PATH)
            >>> hasattr(editor, "hoge_image_arr") and hasattr(editor, "hoge_image_pil")
            True
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"{toBLUE(path)} is not found.")
        img_pil = Image.open(path)
        width, height = img_pil.size
        self.set_attribute(name="pil", value=img_pil)
        self.set_attribute(name="arr", value=cv2.imread(path))
        self.set_attribute(name="width", value=width, msg="")
        self.set_attribute(name="height", value=height, msg="")

        for v, n in zip(
            *assign_trbl(
                data=dict(margin=margin),
                name="margin",
                default=margin_default,
                ret_name=True,
            )
        ):
            self.set_attribute(name=n, value=v, msg="")

    def set_locations(
        self,
        top: Optional[Union[BaseElement, int]] = None,
        right: Optional[Union[BaseElement, int]] = None,
        left: Optional[Union[BaseElement, int]] = None,
        bottom: Optional[Union[BaseElement, int]] = None,
    ) -> None:
        """Automatically calculate and find the optimal locations for both ``left`` and ``top``

        Args:
            top (Optional[Union[BaseElement, int]], optional)    : Reference element or absolute value at the top. Defaults to ``None``.
            right (Optional[Union[BaseElement, int]], optional)  : Reference element or absolute value at the right. Defaults to ``None``.
            left (Optional[Union[BaseElement, int]], optional)   : Reference element or absolute value at the left. Defaults to ``None``.
            bottom (Optional[Union[BaseElement, int]], optional) : Reference element or absolute value at the bottom. Defaults to ``None``.
        """
        self.set_location(lb=top, ub=bottom, direction="vertical")
        self.set_location(lb=left, ub=right, direction="horizontal")

    def set_location(
        self,
        lb: Optional[Union[BaseElement, int]] = None,
        ub: Optional[Union[BaseElement, int]] = None,
        direction: str = "vertical",
        ratio: Tuple[Number, Number] = (1, 1),
    ) -> None:
        """Automatically calculate and find the optimal location (``left`` or ``top``)

        Args:
            lb (Optional[Union[BaseElement, int]], optional) : Lower bound of location. Defaults to ``None``.
            ub (Optional[Union[BaseElement, int]], optional) : Upper bound of location. Defaults to ``None``.
            direction (str, optional)                        : Direction of ``lb`` and ``ub`` line up. Please choose from ``"vertical"`` or ``"horizontal"``. Defaults to ``"vertical"``.
            ratio (Tuple[Number,Number], optional)           : If ``lb`` and ``ub`` are both instances of :class:`BaseElement <veditor.elements.base.BaseElement>`, at what ratio do you split between the 2 elements? Defaults to ``(1, 1)``.
        """
        if direction.lower().startswith("v"):
            lb_name, ub_name, size_name = ("top", "bottom", "height")
        else:
            lb_name, ub_name, size_name = ("left", "right", "width")
        if lb is None:
            if ub is None:
                self.logger.error(
                    f"Couldn't find the location of {toGREEN(lb_name)} and {toGREEN(ub_name)}. Please specify either {toBLUE(lb_name)} or {toBLUE(ub_name)}."
                )
                lb: int = 0 + getattr(self, f"margin_{lb_name}")
            else:
                if isinstance(ub, BaseElement):
                    ub: int = (
                        getattr(ub, lb_name)
                        - getattr(ub, f"margin_{lb_name}")
                        - getattr(self, f"margin_{ub_name}")
                    )
                lb: int = ub - getattr(self, size_name)
        else:
            if isinstance(lb, BaseElement):
                lb: int = (
                    getattr(lb, ub_name)
                    + getattr(lb, f"margin_{ub_name}")
                    + getattr(self, f"margin_{lb_name}")
                )
            if ub is not None:
                if isinstance(ub, BaseElement):
                    ub: int = ub.lb - ub.margin_lb - getattr(self, f"margin_{ub_name}")
                lb = lb + int(
                    (ub - lb - getattr(self, size_name)) / sum(ratio) * ratio[0]
                )
        self.set_attribute(name=lb_name, value=lb, msg="")

    def show_image_arr(self, ax: Optional[Axes] = None) -> Axes:
        """Show ``image_arr`` using :func:`cv2plot <veditor.utils.image_utils.cv2plot>`

        Args:
            ax (Optional[Axes], optional) : An ``Axes`` instance. Defaults to ``None``.

        Returns:
            Axes: An ``Axes`` instance with ``frame`` drawn.
        """
        ax = cv2plot(frame=self.arr, ax=ax, isBGR=True)
        return ax

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
                    self.logger.error(
                        f"Please specify at least one of {toBLUE('dsize')}, {toBLUE('width')}, {toBLUE('height')}."
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
        self.set_attribute(name="width", value=width, msg="")
        self.set_attribute(name="height", value=height, msg="")
