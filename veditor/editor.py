# coding: utf-8
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image
from tqdm import tqdm

from .elements import BaseElement
from .utils._colorings import toBLUE, toGREEN
from .utils._loggers import get_logger
from .utils.audio_utils import synthesize_audio
from .utils.image_utils import arr2pil, pil2arr
from .utils.video_utils import capture2writor


class VEditor(BaseElement):
    def __init__(
        self,
        elements: List[BaseElement] = [],
        width: Optional[int] = None,
        height: Optional[int] = None,
        bgRGB: Optional[Tuple[int, int, int]] = (0, 0, 0),
    ):
        self.elements = elements
        super().__init__(pos_frames=(None, None))
        self.set_element_attributes(bgRGB=bgRGB)

    def set_element_attributes(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        bgRGB: Optional[Tuple[int, int, int]] = (0, 0, 0),
    ) -> None:
        self.set_attribute(name="_width", value=width)
        self.set_attribute(name="_height", value=height)
        self.set_attribute(name="bgRGB", value=bgRGB)
        self.set_pos_frames()
        self.set_trbl()

    def set_trbl(self):
        w = self._width
        h = self._height
        top, right, bottom, left = (1e5, -1, -1, 1e5)
        for element in self.elements:
            t, r, b, l = element.locations
            if t < top:
                top = t
            if r > right:
                right = r
            if b > bottom:
                bottom = b
            if l < left:
                left = l
        if w is None:
            w = r - l
        if h is None:
            h = b - t
        self.set_size(width=w, height=h)
        self.set_locations(top=top, right=right, left=left, bottom=bottom)

    def set_pos_frames(self) -> None:
        start_pos, end_pos = (None, None)
        for element in self.elements:
            s = element.start_pos
            e = element.end_pos
            if (start_pos is None) or ((s is not None) and (s < start_pos)):
                start_pos = s
            if (end_pos is None) or ((e is not None) and (e > end_pos)):
                end_pos = e
        self.set_attribute(name="start_pos", value=start_pos)
        self.set_attribute(name="end_pos", value=end_pos)

    def append(self, element: BaseElement) -> None:
        """Append a new :class:`element <veditor.elements.base.BaseElement>`

        Args:
            element (BaseElement) : An instance of  :class:`BaseElement <veditor.elements.base.BaseElement>`.
        """
        self.elements.append(element)
        self.set_pos_frames()
        self.set_trbl()

    def edit(self, frame: npt.NDArray[np.uint8], pos: int) -> npt.NDArray[np.uint8]:
        """Edit a ``pos``-th frame in the video ``vide_path``.

        Args:
            frame (npt.NDArray[np.uint8]) : The current frame (BGR image) in the video.
            pos (int)                     : The current position in the video.

        Returns:
            npt.NDArray[np.uint8]: An editied frame.
        """
        if self.bgRGB is not None:
            frame[self.top : self.bottom, self.left : self.right, :] = np.full(
                shape=(self.height, self.width, 3),
                fill_value=self.bgRGB,
                dtype=np.uint8,
            )
        for element in self.elements:
            frame = element.edit(frame=frame, pos=pos)
        return frame

    def check_work(
        self,
        pos: int,
        frame: Optional[npt.NDArray[np.uint8]] = None,
        as_pil: bool = True,
    ) -> Union[npt.NDArray[np.uint8], Image.Image]:
        """Check the editing result for ``pos`` frame in video at ``video_path`` of this editor.

        Args:
            pos (int)                                         : The position in the video.
            frame (Optional[npt.NDArray[np.uint8]], optional) : [description]. Defaults to ``None``
            as_pil (bool, optional)                           : Whether to return object as ``Image.Image`` or ``npt.NDArray[npt.uint8]``. Defaults to ``True``.

        Raises:
            ValueError: When ``bgRGB`` is not set, and ``frame`` is ``None``.

        Returns:
            Union[npt.NDArray[np.uint8], Image.Image]: An editing result for the ``pos``-th frame.
        """
        if frame is None:
            if self.bgRGB is None:
                raise ValueError(
                    f"If background color {toGREEN('bgRGB')} is not set, please specify an argument {toBLUE('frame')}."
                )
            else:
                frame = np.full(
                    shape=(self.height, self.width, 3),
                    fill_value=self.bgRGB,
                    dtype=np.uint8,
                )
        frame = self.edit(frame=frame, pos=pos)
        if as_pil:
            return arr2pil(frame)
        else:
            return frame

    def check_works(
        self,
        out_path: Optional[str] = None,
        codec: str = "H264",
        fps: Optional[float] = None,
        open: bool = True,
        **kwargs,
    ) -> str:
        """Check the editing results of this editor.

        Args:
            out_path (Optional[str], optional) : Path to the created video. Defaults to ``None``.
            codec (str, optional)              : Video codec for the created video. Defaults to ``"H264"``.
            fps (Optional[float], optional)    : Frame rate of the output video. Defaults to ``None``.
            open (bool, optional)              : Whether to open output file or not. Defaults to ``True``.

        Returns:
            str: The path to the created video.
        """
        return self.export(out_path=out_path, codec=codec, fps=fps, open=open, **kwargs)

    def export(
        self,
        out_path: Optional[str] = None,
        codec: str = "H264",
        fps: Optional[float] = None,
        open: bool = True,
        **kwargs,
    ) -> str:
        """Create a video with each element in ``elements``.

        Args:
            out_path (Optional[str], optional) : Path to the output video. Defaults to ``None``.
            codec (str, optional)              : Video codec for the output video. Defaults to ``"H264"``.
            fps (Optional[float], optional)    : Frame rate of the output video. Defaults to ``None``.
            open (bool, optional)              : Whether to open the created video file. Defaults to ``True``.

        Returns:
            str: The path to the created video file.
        """
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        out, out_path = capture2writor(
            cap=cap,
            out_path=out_path,
            codec=codec,
            H=self.height,
            W=self.width,
            fps=fps,
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
