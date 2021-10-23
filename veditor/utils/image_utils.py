# coding: utf-8
import math
import string
import textwrap
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from PIL import Image, ImageDraw, ImageFont

from .generic_utils import assign_trbl, flatten_dual, handleKeyError


def arr2pil(frame: npt.NDArray[np.uint8]) -> Image.Image:
    """Convert from ``frame`` (BGR ``npt.NDArray``) to ``image`` (RGB ``Image.Image``)

    Args:
        frame (npt.NDArray[np.uint8]) : A BGR ``npt.NDArray``.

    Returns:
        Image.Image: A RGB ``Image.Image``
    """
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def pil2arr(image: Image.Image) -> npt.NDArray[np.uint8]:
    """Convert from ``image`` (RGB ``Image.Image``) to ``frame`` (BGR ``npt.NDArray``).

    Args:
        image (Image.Image) : A RGB ``Image.Image``

    Returns:
        npt.NDArray[np.uint8] : A BGR ``npt.NDArray``.
    """
    return cv2.cvtColor(np.asarray(image.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2BGR)


def cv2plot(frame: npt.NDArray[np.uint8], ax: Optional[Axes] = None, isBGR: bool = True) -> Axes:
    """Plot a ``frame``.

    Args:
        frame (npt.NDArray[np.uint8]) : Input image.
        ax (Optional[Axes], optional) : An ``Axes`` instance. Defaults to ``None``.
        isBGR (bool, optional)        : Whether ``frame`` is BGR (OpenCV format) or not. Defaults to ``True``.

    Returns:
        Axes: An ``Axes`` instance with ``frame`` drawn.
    """
    if ax is None:
        _, ax = plt.subplots()
    if isBGR:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.imshow(X=frame)
    ax.axis("off")
    return ax


def draw_text_in_pil(
    text: str,
    ttfontname: str,
    img: Optional[Image.Image] = None,
    xy: Tuple[int, int] = (0, 0),
    center: Optional[Tuple[int, int]] = None,
    img_size: Tuple = (640, 360),
    img_mode: str = "RGB",
    bgRGB: Union[str, Tuple] = "white",
    textRGB: Union[str, Tuple] = "black",
    fontsize: int = 16,
    fontheight: Optional[int] = None,
    line_height: Optional[int] = None,
    anchor: Optional[str] = None,
    spacing: int = 4,
    align: str = "left",
    direction: str = "ltr",
    features: Optional[str] = None,
    language: Optional[str] = None,
    stroke_width: int = 0,
    stroke_fill: Optional[Union[str, Tuple]] = None,
    embedded_color: bool = False,
    wrap_text: bool = False,
    text_width: Optional[int] = None,
    margin_right: int = 0,
    drop_whitespace: bool = False,
    ret_position: str = "word",
) -> Image.Image:
    """Draw text in pillow image (``Image.Image``).

    Args:
        text (str)                                          : Text to be drawn to ``img``.
        ttfontname (str)                                    : A filename or file-like object containing a TrueType font.
        img (Optional[Image.Image], optional)               : The image to draw in. If this argment is ``None``, img will be created using ``img_size`` and ``bgRGB`` arguments. Defaults to ``None``.
        xy (Tuple, optional)                                : Where to write the ``text``. This value means the coordinates of (``x``, ``y``). Defaults to ``(0, 0)``.
        img_size (Tuple, optional)                          : The image size. Defaults to ``(640, 360)``.
        img_mode (str, optional)                            : Optional mode to use for color values. Defaults to ``"RGB"``.
        bgRGB (Union[str, Tuple], optional)                 : The color of background image. Defaults to ``"white"``.
        textRGB (Union[str, Tuple], optional)               : The color of text. Defaults to ``"black"``.
        fontsize (int, optional)                            : The font size. Defaults to ``16``.
        fontheight (Optional[int], optional)                : The font height. (If not given, automatically calculated.) Defaults to ``None``.
        line_height (Optional[int], optional)               : The line height. (If not given, automatically calculated.) Defaults to ``None``.
        anchor (Optional[str], optional)                    : The text anchor alignment. Defaults to ``None``.
        spacing (int, optional)                             : The number of pixels between lines. Defaults to ``4``.
        align (str, optional)                               : Determines the relative alignment of lines. Defaults to ``"left"``.
        direction (str, optional)                           : Direction of the text. Defaults to ``"ltr"``.
        features (Optional[str], optional)                  : A list of OpenType font features to be used during text layout. Defaults to ``None``.
        language (Optional[str], optional)                  : Language of the text. Defaults to ``None``.
        stroke_width (int, optional)                        : The width of the text stroke. Defaults to ``0``.
        stroke_fill (Optional[Union[str, Tuple]], optional) : Color to use for the text stroke. If not given, will default to the ``textRGB`` parameter. Defaults to ``None``.
        embedded_color (bool, optional)                     : Whether to use font embedded color glyphs (COLR, CBDT, SBIX). Defaults to ``False``.
        wrap_text (bool, optional)                          : Whether to wrap ``text`` for multilines or not. Defaults to ``False``.
        text_width (Optional[int], optional)                : The length of characters in one line. Use it if you specify ``wrap_text==True``. Defaults to ``None``.
        margin_right (int, optional)                        : Right margin of text in image. Use it if you specify ``wrap_text==True``. Defaults to ``0``.
        drop_whitespace (bool, optional)                    : If ``True``, whitespace at the beginning and ending of every line. Defaults to ``False``.
        ret_position (str, optional)                        : Type of the position of next text to be returned. Please choose from ``["line", "word"]``. Defaults to ``"word"``.

    Returns:
        Image.Image: ``img`` with ``text`` drawn.
    """
    handleKeyError(lst=["word", "line"], ret_position=ret_position)

    kwargs = locals()
    if kwargs.pop("center") is not None:
        w, h = check_font_size(**kwargs)
        cx, cy = center
        font = ImageFont.truetype(font=ttfontname, size=int(fontsize))
        _, fh = font.getsize(text)
        xy = (cx - w // 2, cy - (h + fh) // 2)

    if img is None:
        img = Image.new(mode=img_mode, size=img_size, color=bgRGB)
    else:
        img_mode = img.mode
    iw, ih = img.size

    font = ImageFont.truetype(font=ttfontname, size=int(fontsize))
    fw, fh = font.getsize(text)
    fw //= len(text)
    fh = fontheight or line_height or fh

    x, mt = xy
    if wrap_text:
        text_width = text_width or (iw - margin_right - x) // fw
        wrapped_lines = flatten_dual(
            [
                textwrap.wrap(text=t, width=text_width, drop_whitespace=drop_whitespace)
                for t in text.split("\n")
            ]
        )
    else:
        wrapped_lines = [text]

    alpha_composition: bool = False
    if len(textRGB) == 4 and img_mode == "RGBA":
        text_canvas = Image.new(mode=img_mode, size=img.size, color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(im=text_canvas, mode=img_mode)
        alpha_composition = True
    else:
        draw = ImageDraw.Draw(im=img, mode=img_mode)

    if len(wrapped_lines) > 0:
        for i, line in enumerate(wrapped_lines):
            y = i * fh + mt
            draw.text(
                xy=(x, y),
                text=line,
                fill=textRGB,
                font=font,
                anchor=anchor,
                spacing=spacing,
                align=align,
                direction=direction,
                features=features,
                language=language,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
                embedded_color=embedded_color,
            )

    if ret_position == "line":
        xy = (x, y + fh)
    elif ret_position == "word":
        xy = (x + font.getsize(line)[0], y)

    if alpha_composition:
        img = alpha_composite(bg=img, paste=text_canvas, box=(0, 0))
    return (img, xy)


def check_font_size(
    text: str, ttfontname: str, img: Optional[Image.Image] = None, xy: Tuple = (0, 0), **kwargs
) -> Tuple[int, int]:
    """Find out the text when writing letters using :func:`draw_text_in_pil <veditor.utils.image_utils.draw_text_in_pil>`.

    Args:
        text (str)                            : Text to be drawn to ``img``.
        ttfontname (str)                      : A filename or file-like object containing a TrueType font.
        img (Optional[Image.Image], optional) : The image to draw in. If this argment is ``None``, img will be created using ``img_size`` and ``bgRGB`` arguments. Defaults to ``None``.
        xy (Tuple, optional)                  : Where to write the ``text``. This value means the coordinates of (``x``, ``y``). Defaults to ``(0, 0)``.

    Returns:
        Tuple[int, int] : Calculated text size (width, height)
    """
    _, xy_ = draw_text_in_pil(text=text, ttfontname=ttfontname, img=img, xy=xy, **kwargs)
    return (xy_[0] - xy[0], xy_[1] - xy[1])


def draw_cross(
    img: Image.Image,
    size: Union[Tuple[int, int], int],
    width: int = 5,
    fill_color: Tuple[int, int, int] = (255, 0, 0, 255),
    outline: Optional[Tuple[int, int, int]] = None,
    color_mode: str = "RGBA",
    margin: int = 0,
    **kwargs
) -> Image.Image:
    """Draw Cross Mark.

    Args:
        img (Image.Image)                                : Pillow Image.
        size (Union[Tuple[int,int],int])                 : Cross mark size. (width,Height.
        width (int, optional)                            : The width of the cross mark. Defaults to ``5``.
        fill_color (Tuple[int,int,int], optional)        : The color in the line. Defaults to ``(255,0,0,255)``.
        outline (Optional[Tuple[int,int,int]], optional) : The color of the edge of the line. Defaults to ``None``.
        color_mode (str, optional)                       : Color Mode (ex. ``"RGBA"`` , ``"P"`` . Defaults to ``"RGBA"``.
        margin (int, optional)                           : Specify the position. Defaults to ``0``.

    Returns:
        Image.Image: [description]

    .. plot::
        :class: popup-img

        >>> from PIL import Image
        >>> from veditor.utils import draw_cross, SampleData, cv2plot, pil2arr
        >>> base = Image.open(SampleData().IMAGE_PATH)
        >>> img_square = draw_cross(img=base, size=200, width=10)
        >>> img_rect = draw_cross(img=base, size=(100,200), width=10, outline=(0,255,0))
        >>> fig, axes = plt.subplots(ncols=3, figsize=(18, 4))
        >>> for ax, img in zip(axes, [base, img_square, img_rect]):
        ...     ax = cv2plot(pil2arr(img), ax=ax)
        >>> fig.show()
    """
    ori_mode = img.mode
    img = img.convert(color_mode)
    draw = ImageDraw.Draw(img)
    W, H = img.size

    kwargs["margin"] = margin
    mt, mr, mb, ml = assign_trbl(data=kwargs, name="margin")
    if hasattr(size, "__len__"):
        sx, sy = size[:2]
    else:
        sx = sy = size
    sx /= 2
    sy /= 2
    angle = math.atan(sy / sx)
    cx = (W - ml - mr) // 2 + ml
    cy = (H - mt - mb) // 2 + mt
    dx = (width / 2) * math.sin(angle)
    dy = (width / 2) * math.cos(angle)

    for xy in [
        (
            (cx - sx - dx, cy - sy + dy),
            (cx + sx - dx, cy + sy + dy),
            (cx + sx + dx, cy + sy - dy),
            (cx - sx + dx, cy - sy - dy),
        ),
        (
            (cx + sx - dx, cy - sy - dy),
            (cx + sx + dx, cy - sy + dy),
            (cx - sx + dx, cy + sy + dy),
            (cx - sx - dx, cy + sy - dy),
        ),
    ]:
        draw.polygon(
            xy=xy,
            fill=fill_color,
            outline=outline,
        )
    return img.convert(ori_mode)


SUPPORTED_CONVERSION_METHODS: List[int] = [
    "",
    "nega",
    "bgr2rgb",
    "gray",
    "heatmap",
    "minmax",
]


def image_conversion(
    frame: npt.NDArray[np.uint8],
    method: str,
    cmap: Union[str, Colormap] = "Pastel1",
) -> npt.NDArray[np.uint8]:
    """Convert image by ``method`` method

    Args:
        frame (npt.NDArray[np.uint8]) : Input image (BGR).
        method (str)                  : How to convert an image.

    Returns:
        npt.NDArray[np.uint8]: Converted image.

    .. plot::
        :class: popup-img

        >>> import cv2
        >>> import matplotlib.pyplot as plt
        >>> from veditor.utils import (
        ...     SUPPORTED_CONVERSION_METHODS,
        ...     cv2plot,
        ...     image_conversion,
        ...     SampleData,
        >>> )
        >>> frame = cv2.imread(SampleData().IMAGE_PATH)
        >>> num_methods = len(SUPPORTED_CONVERSION_METHODS)
        >>> ncols = 3; nrows = num_methods//ncols
        >>> fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6 * ncols, 4 * nrows))
        >>> for i,method in enumerate(SUPPORTED_CONVERSION_METHODS):
        ...     ax = cv2plot(image_conversion(frame, method=method), ax=axes[i%2][i//2])
        ...     ax.set_title(method)
        >>> fig.show()
    """
    handleKeyError(lst=SUPPORTED_CONVERSION_METHODS, method=method)
    if method == "nega":
        frame = nega_conversion(frame)
    elif method == "bgr2rgb":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif method == "gray":
        frame = np.tile(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, :, None], reps=3)
    elif method == "heatmap":
        frame = apply_heatmap(frame=frame, cmap=cmap)
    elif method == "minmax":
        frame = min_max_normalization(frame=frame)
    return frame


def min_max_normalization(
    frame: npt.NDArray[np.uint8], axis: Optional[int] = None
) -> npt.NDArray[np.uint8]:
    """Perform the following min-max normalization.

    .. math::

        x^{\prime} = \\frac{x - \min(x)}{\max(x) - \min(x)}

    Args:
        frame (npt.NDArray[np.uint8])  : Input image (BGR).
        axis (Optional[int], optional) : Axis or axes along which to operate. Defaults to ``None``.

    Returns:
        npt.NDArray[np.uint8]: Min-max normalized frame. (0 ~ 255)

    .. plot::
        :class: popup-img

        >>> import cv2
        >>> import matplotlib.pyplot as plt
        >>> from veditor.utils import cv2plot, SampleData, min_max_normalization
        >>> frame = cv2.imread(SampleData().IMAGE_PATH)
        >>> fig, axes = plt.subplots(ncols=2, figsize=(6*2, 4))
        >>> ax = cv2plot(frame, ax=axes[0])
        >>> ax.set_title("Original")
        >>> ax = cv2plot(min_max_normalization(frame=frame), ax=axes[1])
        >>> ax.set_title("min-max Normalization")
        >>> fig.show()
    """
    min = np.min(a=frame, axis=axis, keepdims=True)
    max = np.max(a=frame, axis=axis, keepdims=True)
    return (255 * (frame.astype(float) - min) / (max - min)).astype(np.uint8)


def apply_heatmap(
    frame: npt.NDArray[np.uint8],
    cmap: Union[str, Colormap] = "Pastel1",
    normalize: bool = True,
) -> npt.NDArray[np.uint8]:
    """Apply heatmap to an input BGR image.

    Args:
        frame (npt.NDArray[np.uint8])         : Input image (BGR).
        cmap (Union[str, Colormap], optional) : An identifier for color maps. Defaults to ``"Pastel1"``.
        normalize (bool, optional)            : Whether to perform :func:`min-max normalization <veditor.utils.image_utils.min_max_normalization>`. Defaults to ``True``.

    Returns:
        npt.NDArray[np.uint8]: [description]

    .. plot::
        :class: popup-img

        >>> import cv2
        >>> import matplotlib.pyplot as plt
        >>> from veditor.utils import cv2plot, SampleData, apply_heatmap
        >>> frame = cv2.imread(SampleData().IMAGE_PATH)
        >>> colormaps = ["Pastel1", "Set1", "tab10", "hsv", "bwr", "Reds"]
        >>> num_methods = len(colormaps)
        >>> ncols = 3; nrows = num_methods//ncols
        >>> fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6 * ncols, 4 * nrows))
        >>> for i,cmap in enumerate(colormaps):
        ...     ax = cv2plot(apply_heatmap(frame, cmap=cmap), ax=axes[i%2][i//2])
        ...     ax.set_title(cmap)
        >>> fig.show()
    """
    cmap = plt.get_cmap(cmap)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if normalize:
        gray = min_max_normalization(gray)
    gray = gray.astype(float) / 255.0
    frame = cv2.cvtColor((255 * cmap(gray)).astype(np.uint8)[:, :, :3], cv2.COLOR_RGB2BGR)
    return frame


def nega_conversion(frame: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Convert images to negative and positive.

    Args:
        frame (npt.NDArray[np.uint8]) : Input image (BGR).

    Returns:
        npt.NDArray[np.uint8]: A nega-posi converted image.

    .. plot::
        :class: popup-img

        >>> import cv2
        >>> import matplotlib.pyplot as plt
        >>> from veditor.utils import cv2plot, SampleData, nega_conversion
        >>> frame = cv2.imread(SampleData().IMAGE_PATH)
        >>> fig, axes = plt.subplots(ncols=2, figsize=(6*2, 4))
        >>> ax = cv2plot(frame, ax=axes[0])
        >>> ax.set_title("Original")
        >>> ax = cv2plot(nega_conversion(frame=frame), ax=axes[1])
        >>> ax.set_title("Nega-posi Conversion")
        >>> fig.show()
    """
    return cv2.bitwise_not(frame)


def alpha_composite(
    bg: Image.Image, paste: Image.Image, box: Tuple[int, int] = (0, 0)
) -> Image.Image:
    """Paste the ``paste`` image to ``bg`` with considering alpha channel.

    Args:
        bg (Image.Image)               : Background Image.
        paste (Image.Image)            : Image to paste.
        box (Tuple[int,int], optional) : Where to paste the ``paste`` image on ``bg`` image. Defaults to ``(0,0)``.

    Returns:
        Image.Image: The alpha-composited image.
    """
    bg = bg.convert("RGBA")
    paste = paste.convert("RGBA")
    img_clear = Image.new(mode="RGBA", size=bg.size, color=(255, 255, 255, 0))
    img_clear.paste(paste, box=box)
    bg = Image.alpha_composite(im1=bg, im2=img_clear)
    return bg
