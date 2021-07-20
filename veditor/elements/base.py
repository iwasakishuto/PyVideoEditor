# coding: utf-8
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from PIL import Image
from tqdm import tqdm

from ..utils._colorings import toBLUE, toGREEN
from ..utils._loggers import get_logger
from ..utils.audio_utils import overlay_audio, synthesize_audio
from ..utils.image_utils import arr2pil, cv2plot, pil2arr
from ..utils.video_utils import capture2writor


class BaseElement(ABC):
    ELEMENT_IDX: int = 0

    def __init__(self, pos_frames: Tuple[int, Optional[int]] = (0, None)):
        self.logger = get_logger(name=self.element_name)
        self.start_pos, self.end_pos = pos_frames
        BaseElement.ELEMENT_IDX += 1

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
            msg = ""
        else:
            msg = str(msg)
            if len(msg) == 0:
                msg = str(value)
            msg = " " + msg
        self.logger.info(f"Set attribute {toGREEN(name)}.{msg}")
        setattr(self, name, value)

    @abstractmethod
    def set_locations(self):
        print("Set locations.")

    @abstractmethod
    def edit(
        self, frame: npt.NDArray[np.uint8], pos: int, **kwargs
    ) -> npt.NDArray[np.uint8]:
        """Edit a ``pos``-th frame in the video ``vide_path``.

        Args:
            frame (npt.NDArray[np.uint8]) : The current frame (BGR image) in the video.
            pos (int)                     : The current position in the video.

        Returns:
            npt.NDArray[np.uint8]: An editied frame.
        """
        if (self.start_pos <= pos) and (
            (self.end_pos is None) or (pos <= self.end_pos)
        ):
            frame = np.zeros_like(shape=frame, dtype=np.uint8)
        return frame

    def create_audio_for_overlay(self) -> Tuple[bool, str]:
        """Create an audio for overlaying."""
        return (False, "")

    def overlayed_audio_create(self, video_path: str, fps: float) -> str:
        is_ok, overlay_media_path = self.create_audio_for_overlay()
        if is_ok:
            overlayed_audio_path = overlay_audio(
                base_media_path=video_path,
                overlay_media_path=overlay_media_path,
                position=int(self.start_pos / fps * 1000),
            )
            self.logger.info(
                f"Overlayed audio file is created at {toBLUE(overlayed_audio_path)}"
            )
            return overlayed_audio_path
        return video_path

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

    def check_works(
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
        """Check the editing results of this editor.

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
        audio_path = self.overlayed_audio_create(video_path=video_path, fps=fps)
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
