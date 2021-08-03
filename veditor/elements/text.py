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
            ),  # kwargs
            **kwargs,
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

    def set_size(
        self,
        ttfontname: str,
        xy: Tuple = (0, 0),
        textRGB: Union[str, Tuple] = "black",
        fontsize: int = 16,
        dsize: Optional[Tuple[int, int]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Set size attributes. (``width``, ``height``)

        Args:
            text (str)                       : [description].
            width (Optional[int], optional)  : [description]. Defaults to ``None``.
            height (Optional[int], optional) : [description]. Defaults to ``None``.
        """
        self.set_text_attributes(
            ttfontname=ttfontname, xy=xy, textRGB=textRGB, fontsize=fontsize, **kwargs
        )
        width, height = self.calc_text_size()
        super().set_size(width=width, height=height)

    def calc_text_size(self, text: Optional[str] = None) -> Tuple[int, int]:
        """Calculate the ``text`` size from attributes of this element.

        Args:
            text (Optional[str], optional) : A text string to write. Defaults to ``None``.

        Returns:
            Tuple[int, int]: A tuple for ``width`` and ``height``.

        Examples:
            >>> from veditor.utils import SampleData
            >>> from veditor.elements import TextElement
            >>> element = TextElement(text="PyVideoEditor", ttfontname=SampleData().FONT_POKEFONT_PATH)
            >>> element.calc_text_size()
            (208, 22)
            >>> element.calc_text_size(text="veditor")
            (112, 22)
        """

        _, (width, _) = self.draw_text(img=None, text=text, ret_position="word")
        _, (_, height) = self.draw_text(img=None, text=text, ret_position="line")
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
        img = arr2pil(frame)
        img, _ = self.draw_text(img)
        frame = pil2arr(img)
        return frame
