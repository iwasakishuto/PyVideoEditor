# coding: utf-8
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image
from tqdm import tqdm

from ..utils._loggers import get_logger
from ..utils.audio_utils import synthesize_audio
from ..utils.video_utils import capture2writor
from .base import BaseElement


class VideoElement(BaseElement):
    def __init__(
        self,
        video_path: str,
        start_pos: int = 0,
        margin: Union[int, List[int]] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        top: Optional[Union[BaseElement, int]] = 0,
        right: Optional[Union[BaseElement, int]] = None,
        left: Optional[Union[BaseElement, int]] = 0,
        bottom: Optional[Union[BaseElement, int]] = None,
    ):
        cap = cv2.VideoCapture(video_path)
        super().__init__(pos_frames=(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        self.set_video_attributes(video_path)

    def set_size(
        self,
        x: Union[str, npt.NDArray[np.uint8]],
        dsize: Optional[Tuple[int, int]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """Set

        Args:
            x (Union[str, npt.NDArray[np.uint8]])       : [description].
            dsize (Optional[Tuple[int, int]], optional) : [description]. Defaults to ``None``.
            width (Optional[int], optional)             : [description]. Defaults to ``None``.
            height (Optional[int], optional)            : [description]. Defaults to ``None``.
        """
        self.set_video_attributes(video_path=self.video_path)
        self.resize(dsize=dsize, width=width, height=height)

    def set_video_attributes(self, video_path: str) -> None:
        cap = cv2.VideoCapture(video_path)
        self.set_attribute(
            name="frame_count", value=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        self.set_attribute(name="width", value=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.set_attribute(name="height", value=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.set_attribute(name="fps", value=cap.get(cv2.CAP_PROP_FPS))
        self.set_attribute(name="video_path", value=video_path)

    def set_fps(self, fps: float) -> None:
        self.set_attribute(name="fps", value=fps, msg=f"Changed fps to {fps}")

    @property
    def video_filename(self) -> str:
        """Final component of a pathname of ``video_path``."""
        return os.path.basename(self.video_path)

    def append(self, element: BaseElement) -> None:
        """Append a new :class:`element <veditor.elements.base.BaseElement>`

        Args:
            element (BaseElement) : An instance of  :class:`BaseElement <veditor.elements.base.BaseElement>`.
        """
        self.elements.append(element)

    def edit(self, frame: npt.NDArray[np.uint8], pos: int) -> npt.NDArray[np.uint8]:
        """Edit a ``pos``-th frame in the video ``vide_path``.

        Args:
            frame (npt.NDArray[np.uint8]) : The current frame (BGR image) in the video.
            pos (int)                     : The current position in the video.

        Returns:
            npt.NDArray[np.uint8]: An editied frame.
        """
        for element in self.elements:
            frame = element.edit(frame=frame, pos=pos)
        return frame

    def check_work(
        self, pos: int, as_pil: bool = True
    ) -> Union[npt.NDArray[np.uint8], Image.Image]:
        """Check the editing result for ``pos`` frame in video at ``video_path`` of this editor.

        Args:
            pos (int)                  : The position in the video.
            as_pil (bool, optional)    : Whether to return object as ``Image.Image`` or ``npt.NDArray[npt.uint8]``. Defaults to ``True``.

        Returns:
            Union[npt.NDArray[np.uint8], Image.Image]: An editing result for the ``pos``-th frame.
        """
        return super().check_work(video_path=self.video_path, pos=pos, as_pil=as_pil)

    def check_works(
        self,
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
        return super().check_works(
            video_path=self.video_path,
            audio_path=audio_path,
            out_path=out_path,
            codec=codec,
            H=H,
            W=W,
            open=open,
            **kwargs,
        )

    def export(
        self,
        out_path: Optional[str] = None,
        codec: str = "H264",
        H: Optional[int] = None,
        W: Optional[int] = None,
        fps: Optional[float] = None,
        open: bool = True,
        **kwargs,
    ) -> str:
        """Create a video with each element in ``elements``.

        Args:
            out_path (Optional[str], optional) : Path to the output video. Defaults to ``None``.
            codec (str, optional)              : Video codec for the output video. Defaults to ``"H264"``.
            H (Optional[int], optional)        : Height of the output video. Defaults to ``None``.
            W (Optional[int], optional)        : Width of the output video. Defaults to ``None``.
            fps (Optional[float], optional)    : Frame rate of the output video. Defaults to ``None``.
            open (bool, optional)              : Whether to open the created video file. Defaults to ``True``.

        Returns:
            str: The path to the created video file.
        """
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        out, out_path = capture2writor(
            cap=cap, out_path=out_path, codec=codec, H=H, W=W, fps=fps
        )
        for i in tqdm(
            range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc=self.video_filename
        ):
            ret, frame = cap.read()
            if (not ret) or (frame is None):
                break
            frame = self.edit(frame=frame, pos=i)
            out.write(frame)
        out.release()
        cap.release()

        return self.synthesize_audio(out_path=out_path, open=open)

    def synthesize_audio(self, out_path: str, open: bool = True) -> str:
        """Create audio with each editor in ``editors`` and attach it to the video at ``out_path``.

        Args:
            out_path (str)        : The path to the output video.
            open (bool, optional) : Whether to open the created video file. Defaults to ``True``.

        Returns:
            str: The path to the created video file.
        """
        audio_path = self.video_path
        for element in self.elements:
            audio_path = element.overlayed_audio_create(
                video_path=audio_path, fps=self.fps
            )
        synthesized_video_path = synthesize_audio(
            video_path=out_path,
            audio_path=audio_path,
            open=open,
        )
        return synthesize_audio

    def overlayed_audio_create(self, video_path: str, fps: float) -> str:
        for element in self.elements:
            video_path = element.overlayed_audio_create(video_path=video_path, fps=fps)
        return video_path
