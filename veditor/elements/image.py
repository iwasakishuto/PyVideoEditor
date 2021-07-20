# coding: utf-8

import os
from numbers import Number
from typing import List, Optional, Tuple, Union

import cv2
from PIL import Image

from ..utils._colorings import toBLUE, toGREEN
from ..utils.generic_utils import assign_trbl
from .base import BaseElement


class ImageElement(BaseElement):
    def __init__(
        self,
        path: str,
        pos_frames: Tuple[int, int],
        margin: Optional[Union[Number, List[Number]]] = None,
    ):
        super().__init__()
        self.set_image_attribute(path, margin=margin)

    def set_image_attribute(
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
        self.set_attribute(name="image_pil", value=img_pil)
        self.set_attribute(name="image_arr", value=cv2.imread(path))
        self.set_attribute(name="image_width", value=width, msg="")
        self.set_attribute(name="image_height", value=height, msg="")

        for v, n in zip(
            *assign_trbl(
                data={"margin": margin},
                name="margin",
                default=margin_default,
                ret_name=True,
            )
        ):
            self.set_attribute(name=f"image_{n}", value=v, msg="")
