# coding: utf-8
import math

from .generic_utils import handleTypeError

SUPPORTED_COLOR_CODES = ["hex", "rgb", "rgba"]


def detect_color_code_type(color):
    """Detect Color Code type

    Args:
        color (tuple / str): color code.

    Examples:
        >>> from pycharmers.utils import detect_color_code_type
        >>> detect_color_code_type("#FFFFFF")
        'hex'
        >>> detect_color_code_type((255,255,255))
        'rgb'
        >>> detect_color_code_type((0,0,0,1))
        'rgba'
    """
    handleTypeError(types=[str, tuple, list], color=color)
    if isinstance(color, str):
        color_code = "hex"
    elif isinstance(color, tuple) or isinstance(color, list):
        color_code = {
            3: "rgb",
            4: "rgba",
        }.get(len(color))
    return color_code


def hex2rgb(hex, max_val=1):
    """Convert color code from ``hex`` to ``rgb``"""
    return tuple(
        [int(hex[-6:][i * 2 : (i + 1) * 2], 16) / 255 * max_val for i in range(3)]
    )


def hex2rgba(hex, max_val=1):
    """Convert color code from ``hex`` to ``rgba``"""
    return rgb2rgba(rgb=hex2rgb(hex=hex[-6:], max_val=max_val), max_val=max_val)


def rgb2hex(rgb, max_val=1):
    """Convert color code from ``rgb`` to ``hex``"""
    return "#" + "".join([format(int(255 / max_val * e), "02x") for e in rgb]).upper()


def rgb2rgba(rgb, max_val=1):
    """Convert color code from ``rgb`` to ``rgba``"""
    return (*rgb, 1)


def rgba2hex(rgba, max_val=1):
    """Convert color code from ``rgba`` to ``hex``"""
    return rgb2hex(rgb=rgba2rgb(rgba=rgba, max_val=max_val), max_val=max_val)


def rgba2rgb(rgba, max_val=1):
    """Convert color code from ``rgba`` to ``rgb``"""
    alpha = rgba[-1]
    rgb = rgba[:-1]
    type_ = int if max_val == 255 else float
    # compute the color as alpha against white
    return tuple([type_(alpha * e + (1 - alpha) * max_val) for e in rgb])


def _do_nothing(color, max_val=1):
    return color


def _toColorCode_create(to_color_code):
    def toColorCode(color, max_val=1):
        color_code = detect_color_code_type(color=color)
        return {
            color_code: globals().get(
                f"{color_code}2{to_color_code.lower()}", _do_nothing
            )
            for color_code in SUPPORTED_COLOR_CODES
        }.get(color_code)(color, max_val=max_val)

    toColorCode.__doc__ = f"""Convert color code to {to_color_code.upper()}

    Args:
        color (tuple / str): color code.

    Examples:
        >>> from pycharmers.utils import to{to_color_code.upper()}
        >>> to{to_color_code.upper()}("#FFFFFF")
        {toColorCode("#FFFFFF")}
        >>> to{to_color_code.upper()}((255, 255, 255), max_val=255)
        {toColorCode((255, 255, 255), max_val=255)}
        >>> to{to_color_code.upper()}((1, 1, 1, 1), max_val=1)
        {toColorCode((1, 1, 1, 1), max_val=1)}

    """
    return toColorCode


toHEX = _toColorCode_create("hex")
toRGB = _toColorCode_create("rgb")
toRGBA = _toColorCode_create("rgba")


def choose_text_color(color, max_val=255, is_bgr=False):
    """Select an easy-to-read text color from the given color.

    Args:
        color (tuple / str) : color code.
        max_val (int)       : Maximum value.

    References:
        `WCAG <https://www.w3.org/TR/WCAG20/#relativeluminancede>`_

    Examples:
        >>> from pycharmers.utils import choose_text_color
        >>> from pycharmers.opencv import (cv2BLACK, cv2RED, cv2GREEN, cv2YELLOW, cv2BLUE, cv2MAGENTA, cv2CYAN, cv2WHITE)
        >>> colors = locals().copy()
        >>> for name,color in colors.items():
        ...     if name.startswith("cv2") and isinstance(color, tuple):
        ...         print(f"{name.lstrip('cv2'):<7}: {str(color):<15} -> {choose_text_color(color=color, max_val=255, is_bgr=True)}")
    """
    color_code = detect_color_code_type(color=color)
    rgb = toRGB(color=color, max_val=max_val)
    if is_bgr:
        rgb = rgb[::-1]

    def sRGB2RGB(e):
        i = e / max_val
        return i / 12.92 if i <= 0.03928 else math.pow((i + 0.055) / 1.055, 2.4)

    R, G, B = [sRGB2RGB(e) for e in rgb]
    # Relative Brightness BackGround.
    Lbg = 0.2126 * R + 0.7152 * G + 0.0722 * B

    Lw = 1  # Relative Brightness of White
    Lb = 0  # Relative Brightness of Black

    Cw = (Lw + 0.05) / (Lbg + 0.05)
    Cb = (Lbg + 0.05) / (Lb + 0.05)
    text_rgb = (0, 0, 0) if Cb > Cw else (max_val, max_val, max_val)
    return {"rgb": _do_nothing, "hex": rgb2hex, "rgba": _do_nothing,}.get(
        color_code
    )(text_rgb, max_val=max_val)


def generate_color_series(color, variation, diff=10, reverse=False):
    """Generate light and dark color series.

    Args:
        color (tuple)   : Color [0,255]
        variation (int) : How many colors to create.
        diff (int)      : How much to change
        reverse (bool)  : If ``True``, sort in descending order.

    Returns:
        colors (list) : colors.

    Examples:
        >>> from pycharmers.utils import generateLightDarks
        >>> generateLightDarks(color=(245,20,25), variation=3, diff=10)
        [(235, 10, 15), (245, 20, 25), (255, 30, 35)]
        >>> generateLightDarks(color=(245, 20, 25), variation=3, diff=-10)
        [(225, 0, 5), (235, 10, 15), (245, 20, 25)]
    """
    val = max(color[:3]) if diff > 0 else min(color[:3])
    u = 0
    for _ in range(variation - 1):
        val += diff
        if not 255 >= val >= 0:
            break
        u += 1
    return sorted(
        [
            tuple(
                [
                    max(min(e + diff * (u - v), 255), 0) if i < 3 else e
                    for i, e in enumerate(color)
                ]
            )
            for v in range(variation)
        ],
        reverse=reverse,
    )
