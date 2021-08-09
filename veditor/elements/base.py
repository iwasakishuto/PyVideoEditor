# coding: utf-8
import os
from abc import ABC, abstractmethod
from numbers import Number
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from PIL import Image
from tqdm import tqdm

from ..utils._colorings import toBLUE, toGREEN
from ..utils._loggers import get_logger
from ..utils.audio_utils import overlay_audio, synthesize_audio
from ..utils.generic_utils import assign_trbl
from ..utils.image_utils import arr2pil, cv2plot, pil2arr
from ..utils.video_utils import capture2writor


class BaseElement(ABC):
    ELEMENT_IDX: int = 0

    def __init__(self, pos_frames: Tuple[int, Optional[int]] = (0, None)):
        self.logger = get_logger(name=self.element_name)
        self.start_pos, self.end_pos = pos_frames
        BaseElement.ELEMENT_IDX += 1

    def __repr__(self):
        return f"{self.element_name} {self.locations}"

    @property
    def element_name(self) -> str:
        return f"{BaseElement.ELEMENT_IDX}.{self.__class__.__name__}"

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def locations(self) -> Tuple[int, int, int, int]:
        return (self.top, self.right, self.bottom, self.left)

    @property
    def trbl(self) -> Tuple[int, int, int, int]:
        return self.locations

    def inCharge(self, pos: int) -> bool:
        """Find out if this element is in charge."""
        return (self.start_pos <= pos) and (
            (self.end_pos is None) or (pos <= self.end_pos)
        )

    def set_attribute(self, name: str, value: str, msg: Optional[str] = None) -> None:
        """Set attribute to this class with logs using ``setattr``.

        Args:
            name (str)          : An attribute name.
            value (str)         : An attribute value.
            msg (str, optional) : Additional log message. Defaults to ``""``.

        Examples:
            >>> from veditor.chaptors import MarqueeEditor
            >>> editor = MarqueeEditor
            >>> editor.set_attribute(name="hoge", value=1)
            >>> hasattr(editor, "hoge")
            True
            >>> editor.hoge
            1
        """
        if msg is None:
            msg = str(value)
        self.logger.info(f"Set attribute {toGREEN(name)}. {msg}")
        setattr(self, name, value)

    @abstractmethod
    def edit(
        self, frame: npt.NDArray[np.uint8], pos: int, **kwargs
    ) -> npt.NDArray[np.uint8]:
        """Edit a ``pos``-th frame.

        Args:
            frame (npt.NDArray[np.uint8]) : The current frame (BGR image) (in the video)
            pos (int)                     : The current position (in the video)

        Returns:
            npt.NDArray[np.uint8]: An editied frame.
        """
        if self.inCharge(pos):
            frame = np.zeros_like(shape=frame, dtype=np.uint8)
        return frame

    def create_audio_for_overlay(self) -> Tuple[bool, str]:
        """Create an audio for overlaying."""
        return (False, "")

    def overlay_audio(self, base_media_path: str, frame_rate: float) -> str:
        """Overlay audio if this :class:`element <veditor.elements.base.BaseElement>` has its own audio.

        Args:
            base_media_path (str) : The path to media file (contains audio) to be overlayed.
            frame_rate (float)    : The frame rate for the media at ``base_media_path``.

        Returns:
            str: The path to created audio file.
        """
        is_ok, overlay_media_path = self.create_audio_for_overlay()
        if is_ok:
            overlayed_audio_path = overlay_audio(
                base_media_path=base_media_path,
                overlay_media_path=overlay_media_path,
                position=int(self.start_pos / frame_rate * 1000),
            )
            self.logger.info(
                f"Overlayed audio file is created at {toBLUE(overlayed_audio_path)}"
            )
            return overlayed_audio_path
        return base_media_path

    def check_work_in_video(
        self, video_path: str, pos: int, as_pil: bool = True
    ) -> Union[npt.NDArray[np.uint8], Image.Image]:
        """Check the editing result for ``pos`` frame in video at ``video_path`` of this editor.

        Args:
            video_path (str)        : The path to the input video file.
            pos (int)               : The position in the video at ``vide_path``
            as_pil (bool, optional) : Whether to return object as ``Image.Image`` or ``npt.NDArray[npt.uint8]``. Defaults to ``True``.

        Returns:
            Union[npt.NDArray[np.uint8], Image.Image]: An edited result for the ``pos``-th frame.
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        is_ok, frame = cap.read()
        if is_ok and (frame is not None):
            frame = self.edit(frame=frame, pos=pos)
            if as_pil:
                frame = arr2pil(frame)
        cap.release()
        return frame

    def check_works_in_video(
        self,
        video_path: str,
        audio_path: Optional[str] = None,
        out_path: Optional[str] = None,
        codec: str = "H264",
        H: Optional[int] = None,
        W: Optional[int] = None,
        fps: Optional[float] = None,
        open: bool = True,
        **kwargs,
    ) -> str:
        """Check the editing results of this editor for a video at ``video_path``.

        Args:
            video_path (str)                   : Path to the input video file.
            audio_path (str, optional)         : Path to the audio file. Defaults to ``None``. (Same as ``video_path``.)
            out_path (Optional[str], optional) : Path to the created video. Defaults to ``None``.
            codec (str, optional)              : Video codec for the created video. Defaults to ``"H264"``.
            H (Optional[int], optional)        : Height of the output video. Defaults to ``None``.
            W (Optional[int], optional)        : Width of the output video. Defaults to ``None``.
            fps (Optional[float], optional)    : Frame rate of the output video. Defaults to ``None``.
            open (bool, optional)              : Whether to open output file or not. Defaults to ``True``.

        Returns:
            str: The path to the created video.
        """
        cap = cv2.VideoCapture(video_path)
        out, out_path = capture2writor(
            cap=cap, out_path=out_path, codec=codec, H=H, W=W, fps=fps
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_pos)
        fps = fps or cap.get(cv2.CAP_PROP_FPS)
        end_pos = self.end_pos or int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        for i in tqdm(range(end_pos - self.start_pos + 1), desc=self.element_name):
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            is_ok, frame = cap.read()
            if (not is_ok) or (frame is None):
                break
            frame = self.edit(frame=frame, pos=pos, **kwargs)
            out.write(frame)
        out.release()
        cap.release()
        # Synthesize Audio.
        audio_path = self.overlay_audio(video_path=video_path, fps=fps)
        out_synthesized_path = synthesize_audio(
            video_path=out_path,
            audio_path=audio_path,
            start=int(1000 * self.start_pos / fps),
            end=int(1000 * end_pos / fps),
            open=open,
            delete_intermidiates=True,
            logger=self.logger,
        )
        return out_synthesized_path


class FixedElement(BaseElement):
    def __init__(
        self,
        pos_frames: Tuple[int, Optional[int]] = (0, None),
        margin: Union[int, List[int]] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        top: Optional[Union[BaseElement, int]] = None,
        right: Optional[Union[BaseElement, int]] = None,
        left: Optional[Union[BaseElement, int]] = None,
        bottom: Optional[Union[BaseElement, int]] = None,
        **kwargs,
    ):
        """Elements with fixed size and location.

        Args:
            pos_frames (Tuple[int, Optional[int]], optional)     : Start and end positions. Defaults to ``(0, None)``.
            margin (Optional[Union[int, List[int]]], optional)   : Margin. Defaults to ``None``.
            width (Optional[int], optional)                      : The element width. Defaults to ``None``.
            height (Optional[int], optional)                     : The element height. Defaults to ``None``.
            top (Optional[Union[BaseElement, int]], optional)    : Reference element or absolute value at the top. Defaults to ``None``.
            right (Optional[Union[BaseElement, int]], optional)  : Reference element or absolute value at the right. Defaults to ``None``.
            left (Optional[Union[BaseElement, int]], optional)   : Reference element or absolute value at the left. Defaults to ``None``.
            bottom (Optional[Union[BaseElement, int]], optional) : Reference element or absolute value at the bottom. Defaults to ``None``.
        """
        super().__init__(pos_frames=pos_frames)
        self.set_margin(margin=margin, margin_default=0)
        width, height = self.calc_element_size(width=width, height=height, **kwargs)
        self.set_size(width=width, height=height)
        self.set_locations(top=top, right=right, left=left, bottom=bottom)

    def calc_element_size(
        self, width: Optional[int] = None, height: Optional[int] = None, **kwargs
    ) -> Tuple[int, int]:
        return (width, height)

    def set_size(
        self, width: Optional[int] = None, height: Optional[int] = None, **kwargs
    ) -> None:
        """Set size attributes (``width`` and ``height``).

        Args:
            width (int)  : [description].
            height (int) : [description].
        """
        if width is None:
            self.logger.error(f"Please specify the {toGREEN('width')}")
        else:
            self.set_attribute(name="width", value=width)
        if height is None:
            self.logger.error(f"Please specify the {toGREEN('height')}")
        else:
            self.set_attribute(name="height", value=height)

    def set_margin(
        self, margin: Optional[Union[int, List[int]]] = None, margin_default: int = 0
    ) -> None:
        """Set an attribute for element margin.

        Args:
            margin (Optional[Union[int, List[int]]], optional) : Margin. Defaults to ``None``.
            margin_default (int, optional)                     : Default value for margin. Defaults to ``0``.
        """
        for v, n in zip(
            *assign_trbl(
                data=dict(margin=margin),
                name="margin",
                default=margin_default,
                ret_name=True,
            )
        ):
            self.set_attribute(name=n, value=v)

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
                    ub: int = (
                        getattr(ub, lb_name)
                        - getattr(ub, f"margin_{lb_name}")
                        - getattr(self, f"margin_{ub_name}")
                    )
                lb = lb + int(
                    (ub - lb - getattr(self, size_name)) / sum(ratio) * ratio[0]
                )
        self.set_attribute(name=lb_name, value=lb)

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
        return frame

    def calc_dsize(
        self,
        dsize: Optional[Tuple[int, int]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Resize the both ``image_pil`` and ``image_arr`` attributes.

        If only ``width`` or ``height`` is given, resize while preserving the aspect ratio.

        Args:
            dsize (Optional[Tuple[int,int]], optional) : Desired image size. Defaults to ``None``.
            width (Optional[int], optional)            : Desired image width. Defaults to ``None``.
            height (Optional[int], optional)           : Desired image height. Defaults to ``None``.

        Returns:
            Tuple[int, int] : Calculated desired size. (``width``, ``height``)
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
        return (width, height)

    def resize(
        self,
        dsize: Optional[Tuple[int, int]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        width, height = self.calc_dsize(dsize=dsize, width=width, height=height)
        self.set_size(width=width, height=height)

    def check_work(
        self, video_path: str, pos: int, as_pil: bool = True
    ) -> Union[npt.NDArray[np.uint8], Image.Image]:
        """Check the editing result for ``pos`` frame in video at ``video_path`` of this editor.

        Args:
            video_path (str)        : The path to the input video file.
            pos (int)               : The position in the video at ``vide_path``
            as_pil (bool, optional) : Whether to return object as ``Image.Image`` or ``npt.NDArray[npt.uint8]``. Defaults to ``True``.

        Returns:
            Union[npt.NDArray[np.uint8], Image.Image]: An edited result for the ``pos``-th frame.
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        is_ok, frame = cap.read()
        if is_ok and (frame is not None):
            frame = self.edit(frame=frame, pos=pos)
            if as_pil:
                frame = arr2pil(frame)
        cap.release()
        return frame
