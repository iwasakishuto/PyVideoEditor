# coding: utf-8
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from PIL import Image

from ..utils.image_utils import arr2pil, draw_text_in_pil, pil2arr
from .base import BaseElement, FixedElement


class TextElement(FixedElement):
    def __init__(
        self,
        text: str,
        ttfontname: str,
        pos_frames: Tuple[int, Optional[int]] = (0, None),
        margin: Union[int, List[int]] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        top: Optional[Union[BaseElement, int]] = None,
        right: Optional[Union[BaseElement, int]] = None,
        left: Optional[Union[BaseElement, int]] = None,
        bottom: Optional[Union[BaseElement, int]] = None,
        xy: Tuple = (0, 0),
        textRGB: Union[str, Tuple] = "black",
        fontsize: int = 16,
        **kwargs,
    ):
        super().__init__(
            pos_frames=pos_frames,
            margin=margin,
            width=width,
            height=height,
            top=top,
            right=right,
            left=left,
            bottom=bottom,
            **dict(
                text=text,
                ttfontname=ttfontname,
                xy=xy,
                textRGB=textRGB,
                fontsize=fontsize,
                **kwargs,
            ),  # kwargs
            **kwargs,
        )
        self.set_text_attributes(
            text=text, ttfontname=ttfontname, fontsize=fontsize, **kwargs
        )

    def set_text_attributes(
        self,
        text: str,
        ttfontname: str,
        xy: Tuple = (0, 0),
        textRGB: Union[str, Tuple] = "black",
        fontsize: int = 16,
        **kwargs,
    ):
        """Set attributes for a text element.

        Args:
            text (str)                            : Text to be drawn to ``img``.
            ttfontname (str)                      : A filename or file-like object containing a TrueType font.
            xy (Tuple, optional)                  : Where to write the ``text``. This value means the coordinates of (``x``, ``y``). Defaults to ``(0, 0)``.
            textRGB (Union[str, Tuple], optional) : The color of text. Defaults to ``"black"``.
            fontsize (int, optional)              : The font size. Defaults to ``16``.

        Examples:
            >>> from veditor.utils import SampleData
            >>> from veditor.elements import TextElement
            >>> element = TextElement(text="PyVideoEditor", ttfontname=SampleData().FONT_POKEFONT_PATH)
            >>> attribute_names = ["text", "ttfontname", "xy", "textRGB", "fontsize"]
            >>> all([hasattr(element, attr) for attr in attribute_names])
            True
        """
        self.set_attribute(name="text", value=text)
        self.set_attribute(name="ttfontname", value=ttfontname)
        self.set_attribute(name="xy", value=xy)
        self.set_attribute(name="textRGB", value=textRGB)
        self.set_attribute(name="fontsize", value=fontsize)
        self.set_attribute(
            name="drawKwargs",
            value=kwargs,
            msg=f"{len(kwargs)} keyword arguments ({', '.join([k for k in kwargs.keys()])}) are set.",
        )

    def calc_element_size(
        self,
        text: str,
        ttfontname: str,
        fontsize: int = 16,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs,
    ) -> Tuple[int, int]:
        drawKwargs = dict(text=text, ttfontname=ttfontname, fontsize=fontsize, **kwargs)
        if width is None:
            drawKwargs.update(dict(img=None, ret_position="word"))
            _, (width, _) = draw_text_in_pil(**drawKwargs)
        if height is None:
            drawKwargs.update(dict(img=None, ret_position="line"))
            _, (_, height) = draw_text_in_pil(**drawKwargs)
        return (width, height)

    def draw_text(
        self, img: Image.Image, text: Optional[str] = None, **kwargs
    ) -> Image.Image:
        """Draw text in pillow image (``Image.Image``) using :func:`draw_text_in_pil <veditor.utils.image_utils.draw_text_in_pil>`.

        Args:
            img (Image.Image)              : An image to which ``text`` is written.
            text (Optional[str], optional) : A text string to write. Defaults to ``None``.

        Returns:
            Image.Image: ``img`` with ``text`` drawn.
        """
        kwargs.update(self.drawKwargs)
        return draw_text_in_pil(
            text=text or self.text,
            ttfontname=self.ttfontname,
            img=img,
            xy=self.xy,
            textRGB=self.textRGB,
            fontsize=self.fontsize,
            **kwargs,
        )

    def edit(
        self, frame: npt.NDArray[np.uint8], pos: int, **kwargs
    ) -> npt.NDArray[np.uint8]:
        """Return ``frame`` as it is.

        Args:
            frame (npt.NDArray[np.uint8]) : The current frame (BGR image) (in the video)
            pos (int)                     : The current position (in the video)

        Returns:
            npt.NDArray[np.uint8]: An editied frame.
        """
        if self.inCharge(pos):
            img = arr2pil(frame)
            img, _ = self.draw_text(img)
            frame = pil2arr(img)
        return frame
