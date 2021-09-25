# coding: utf-8
import math
import os
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm

from ._colorings import toGREEN
from .generic_utils import now_str


def createVideoWritor(
    H: int, W: int, fps: float, codec: str = "avc1", out_path: Optional[str] = None
) -> Tuple[cv2.VideoWriter, str]:
    """Create an instance of ``cv2.VideoWritor``.

    Args:
        H (int)                            : Height of the output video.
        W (int)                            : Width of the output video.
        fps (float)                        : Frame rate of the output video.
        codec (str, optional)              : Video codec for the output video. Defaults to ``"avc1"``.
        out_path (Optional[str], optional) : [description]. Defaults to ``None``.

    Returns:
        Tuple[cv2.VideoWriter, str]: Tuple of ``cv2.VideoWriter`` and path to output video.
    """
    if out_path is None:
        out_path = now_str() + ".mp4"
    out = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    return (out, out_path)


def capture2writor(
    cap: cv2.VideoCapture,
    out_path: Optional[str] = None,
    codec: str = "avc1",
    H: Optional[int] = None,
    W: Optional[int] = None,
    fps: Optional[float] = None,
) -> Tuple[cv2.VideoWriter, str]:
    """Create a suitable ``cv2.VideoWriter`` for input ``cv2.VideoCapture``.

    Args:
        cap (cv2.VideoCapture)             : An instance of ``cv2.VideoCaputure``.
        out_path (Optional[str], optional) : Path to the output video. Defaults to ``None``.
        codec (str, optional)              : Video codec for the output video. Defaults to ``"avc1"``.
        H (Optional[int], optional)        : Height of the output video. Defaults to ``None``.
        W (Optional[int], optional)        : Width of the output video. Defaults to ``None``.
        fps (Optional[float], optional)    : Frame rate of the output video. Defaults to ``None``.

    Returns:
        Tuple[str,cv2.VideoWriter]: Tuple of ``cv2.VideoWriter`` and path to output video.

    Examples:
        >>> import cv2
        >>> from veditor.utils import capture2writor, SampleData
        >>> cap = cv2.VideoCapture(SampleData().VIDEO_PATH)
        >>> out, out_path = capture2writor(cap)
        >>> isinstance(out, cv2.VideoWriter) and isinstance(out_path, str)
        True
    """
    return createVideoWritor(
        H=H or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        W=W or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        fps=fps or cap.get(cv2.CAP_PROP_FPS),
        fourcc=cv2.VideoWriter_fourcc(*codec),
        out_path=out_path,
    )


def show_frames(
    video: Union[str, cv2.VideoCapture],
    start: int = 0,
    end: Optional[int] = None,
    step: int = 1,
    ncols: int = 6,
    nframes: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
    fig: Optional[Figure] = None,
) -> Figure:
    """Cut out frames from the ``video`` and plot them.

    Args:
        video (Union[str, cv2.VideoCapture])          : Path to video or an instance of ``cv2.VideoCaputure``.
        start (int, optional)                         : Draw subsequent frames from ``start``. Defaults to ``0``.
        end (Optional[int], optional)                 : Draw up to ``end``-th frame. If not specified, draw to the end. Defaults to ``None``.
        ncols (int, optional)                         : Number of images lined up side by side (number of columns). Defaults to ``6``.
        figsize (Optional[Tuple[int, int]], optional) : Size of one image. Defaults to ``None``.
        fig (Optional[Figure], optional)              : Figure instance you want to draw in. Defaults to ``None``.

    Returns:
        Figure: Figure where frames from ``start`` to ``end`` are drawn.

    Raises:
        ValueError: When ``video`` is ``cv2.VideoCapture`` and is not Opened.

    .. plot::
      :class: popup-img

        >>> from veditor.utils import show_frames, SampleData
        >>> fig = show_frames(video=SampleData().VIDEO_PATH, step=300, ncols=2)
        >>> fig.show()
    """
    if isinstance(video, cv2.VideoCapture):
        if not video.isOpened():
            raise ValueError(
                f"{toGREEN('video')} is not opened. Please reinitialize the {toGREEN('cv2.VideoCapture')} instance."
            )
        cap = video
    else:
        cap = cv2.VideoCapture(video)
    count = nframes or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    digit = len(str(count))
    end = min(end or count, count)
    nfigs = math.ceil((end - start + 1) / step)
    nrows = (nfigs - 1) // ncols + 1
    # Calculate the appropriate figure size.
    if figsize is None:
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if w < h:
            figsize = (4, 4 * (h / w))
        else:
            figsize = (4 * (w / h), 4)
    if fig is None:
        fig = plt.figure(figsize=(int(figsize[0] * ncols), int(figsize[1] * nrows)))
    counter = 0
    for i in tqdm(range(count), desc=f"step:{step}"):
        ret, frame = cap.read()
        if (not ret) or (frame is None):
            break
        if start <= i <= end:
            if counter % step == 0:
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                ax = fig.add_subplot(nrows, ncols, (i - start) // step + 1)
                ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ax.axis("off")
                ax.set_title(f"No.{i:>0{digit}}/{count}\n{msec/1000:.2f}[s]")
            counter += 1
        elif i > end:
            break
    cap.release()
    return fig


def save_frames(
    video: Union[str, cv2.VideoCapture],
    positions: Union[int, List[int]],
    fmt: str = "{pos}.png",
) -> None:
    """Cut out frames from the ``video`` and save them.

    Args:
        video (Union[str, cv2.VideoCapture]) : Path to video or an instance of ``cv2.VideoCaputure``.
        positions (Union[int, List[int]])    : Which position(s) to save the frame.
        fmt (str, optional)                  : File name format. Must include ``"{pos}"``. Defaults to ``"{pos}.png"``.

    Raises:
        ValueError: When ``video`` is ``cv2.VideoCapture`` and is not Opened, or ``fmt`` DO NOT include ``"{pos}"``.
    """
    if isinstance(video, cv2.VideoCapture):
        if not video.isOpened():
            raise ValueError(
                f"{toGREEN('video')} is not opened. Please reinitialize the {toGREEN('cv2.VideoCapture')} instance."
            )
        cap = video
    else:
        cap = cv2.VideoCapture(video)
    if isinstance(positions, int):
        positions = [positions]
    if "{pos}" not in fmt:
        raise ValueError(f"{toGREEN('fmt')} must include " + '"{pos}"')
    max_pos = max(positions)
    counter = 0
    with tqdm(range(max_pos + 1)) as pbar:
        for i in pbar:
            pbar.set_description(f"saved {counter} frames")
            ret, frame = cap.read()
            if (not ret) or (frame is None):
                break
            if i in positions:
                cv2.imwrite(fmt.format(pos=i), frame)
                counter += 1
    cap.release()


def vcodec2ext(*codec) -> str:
    """Convert video codec to video extension.

    Args:
        codec (Union[tuple, str]) : Video Codec.

    Returns:
        str: Ideal file extension.

    Examples:
        >>> from pycharmers.opencv import vcodec2ext
        >>> vcodec2ext("MP4V")
        '.mp4'
        >>> vcodec2ext("mp4v")
        '.mov'
        >>> vcodec2ext("VP80")
        '.webm'
        >>> vcodec2ext("XVID")
        '.avi
        >>> vcodec2ext("☺️")
        '.mp4'

    Raises:
        KeyError: When the file extension cannot be inferred from the video ``codec``.
    """
    if len(codec) == 1 and isinstance(codec[0], str):
        codec = codec[0]
    else:
        codec = "".join(codec)
    codec2ext = {
        "VP80": ".webm",
        "MP4S": ".mp4",
        "MP4V": ".mp4",
        "mp4v": ".mov",
        "H264": ".mp4",
        "X264": ".mp4",
        "DIV3": ".avi",
        "DIVX": ".avi",
        "IYUV": ".avi",
        "MJPG": ".avi",
        "XVID": ".avi",
        "THEO": ".ogg",
        "H263": ".wmv",
        "avc1": ".mp4",
    }
    handleKeyError(lst=list(codec2ext.keys()), codec=codec)
    return codec2ext[codec]
