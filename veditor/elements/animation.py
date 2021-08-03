# coding: utf-8
import math
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image
from tqdm import tqdm

from ..utils.audio_utils import synthesize_audio
from ..utils.image_utils import alpha_composite, arr2pil, pil2arr
from ..utils.video_utils import capture2writor
from .base import BaseElement, FixedElement


class AnimationElement(FixedElement):
    def __init__(
        self,
        animation_path: str,
        pos_frames: Tuple[int, Optional[int]] = (0, None),
        period: Optional[int] = None,
        margin: Union[int, List[int]] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        top: Optional[Union[BaseElement, int]] = 0,
        right: Optional[Union[BaseElement, int]] = None,
        left: Optional[Union[BaseElement, int]] = 0,
        bottom: Optional[Union[BaseElement, int]] = None,
    ):
        cap = cv2.VideoCapture(animation_path)
        super().__init__(
            pos_frames=pos_frames,
            margin=margin,
            width=width,
            height=height,
            top=top,
            right=right,
            left=left,
            bottom=bottom,
            **dict(animation_path=animation_path),  # kwargs
        )
        self.set_animation_attributes(animation_path=animation_path, period=period)

    def calc_element_size(
        self,
        animation_path: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs,
    ) -> Tuple[int, int]:
        cap = cv2.VideoCapture(animation_path)
        if width is None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if height is None:
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def set_animation_attributes(
        self, animation_path: str, period: Optional[int] = None
    ) -> None:
        cap = cv2.VideoCapture(animation_path)
        img = Image.open(animation_path)
        self.set_attribute(
            name="arr_frame_count", value=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        self.set_attribute(name="pil_frame_count", value=int(img.n_frames))
        self.set_attribute(name="animation_path", value=animation_path)
        self.set_attribute(name="period", value=period)
        self.set_attribute(name="mode", value=img.mode)

    def get_pos_pil(self, pos: int) -> Image.Image:
        img = Image.open(self.animation_path)
        img.seek(
            math.floor(
                math.modf((pos - self.start_pos) / self.period)[0] * self.frame_count
            )
        )
        return img

    def get_pos_arr(self, pos: int) -> npt.NDArray[np.uint8]:
        cap = cv2.VideoCapture(self.animation_path)
        cap.set(
            math.floor(
                math.modf((pos - self.start_pos) / self.period)[0] * self.frame_count
            )
        )
        ret, frame = cap.read()
        return frame

    def edit(self, frame: npt.NDArray[np.uint8], pos: int) -> npt.NDArray[np.uint8]:
        """Edit a ``pos``-th frame in the video ``vide_path``.

        Args:
            frame (npt.NDArray[np.uint8]) : The current frame (BGR image) in the video.
            pos (int)                     : The current position in the video.

        Returns:
            npt.NDArray[np.uint8]: An editied frame.
        """
        if self.inCharge(pos):
            if self.mode == "RGBA":
                img = arr2pil(frame)
                paste = self.get_pos_pil(pos)
                img = alpha_composite(bg=img, paste=paste, box=(self.left, self.top))
                frame = pil2arr(img)
            else:
                arr = self.get_pos_arr(pos)
                frame[self.top : self.bottom, self.left : self.right, :] = arr
        return frame
