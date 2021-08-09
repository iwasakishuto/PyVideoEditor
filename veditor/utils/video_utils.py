# coding: utf-8
import math
import os
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm

from ._colorings import toGREEN
from .generic_utils import now_str


def capture2writor(
    cap: cv2.VideoCapture,
    out_path: Optional[str] = None,
    codec: str = "H264",
    H: Optional[int] = None,
    W: Optional[int] = None,
    fps: Optional[float] = None,
) -> Tuple[str, cv2.VideoWriter]:
    """Create a suitable ``cv2.VideoWriter`` for input ``cv2.VideoCapture``.

    Args:
        cap (cv2.VideoCapture)             : An instance of ``cv2.VideoCaputure``.
        out_path (Optional[str], optional) : Path to the output video. Defaults to ``None``.
        codec (str, optional)              : Video codec for the output video. Defaults to ``"H264"``.
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
        >>> isinstance(out, cv2.VideoWriter)
        True
    """
    W = W or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = H or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = fps or cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    if out_path is None:
        out_path = now_str() + ".mp4"
    out = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    return (out, out_path)


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
    """Cut out frames from the video and plot them.

    Args:
        video (Union[str, cv2.VideoCapture])          : Path to video or an instance of ``cv2.VideoCaputure``.
        start (int, optional)                         : Draw subsequent frames from ``start``. Defaults to ``0``.
        end (Optional[int], optional)                 : Draw up to ``end``-th frame. If not specified, draw to the end. Defaults to ``None``.
        ncols (int, optional)                         : Number of images lined up side by side (number of columns). Defaults to ``6``.
        figsize (Optional[Tuple[int, int]], optional) : Size of one image. Defaults to ``None``.
        fig (Optional[Figure], optional)              : Figure instance you want to draw in. Defaults to ``None``.

    Returns:
        Figure: Figure where frames from ``start`` to ``end`` are drawn.

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
