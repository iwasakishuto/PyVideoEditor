# coding: utf-8
import copy
import math
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from PIL import Image
from tqdm import tqdm

from ..utils.audio_utils import synthesize_audio
from ..utils.image_utils import alpha_composite, arr2pil, pil2arr
from ..utils.video_utils import capture2writor, show_frames
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
        arr_images = self.get_all_arr(animation_path=animation_path)
        pil_images = self.get_all_pil(animation_path=animation_path)
        self.set_attribute(
            name="arr_images",
            value=arr_images,
            msg=f"{len(arr_images)} images were saved.",
        )
        self.set_attribute(
            name="pil_images",
            value=pil_images,
            msg=f"{len(pil_images)} images were saved.",
        )
        self.set_attribute(name="animation_path", value=animation_path)
        self.set_attribute(name="period", value=period or self.pil_frame_count)
        self.set_attribute(name="mode", value=pil_images[-1].mode)

    @property
    def arr_frame_count(self):
        return len(self.arr_images)

    @property
    def pil_frame_count(self):
        return len(self.pil_images)

    def get_all_pil(self, animation_path: str) -> List[Image.Image]:
        img = Image.open(animation_path)
        pil_images = [img]
        for i in range(img.n_frames - 1):
            try:
                img.seek(img.tell() + 1)
                pil_images.append(copy.deepcopy(img))
            except EOFError:
                break
        return pil_images

    def get_all_arr(self, animation_path: str) -> List[npt.NDArray[np.uint8]]:
        arr_images = []
        cap = cv2.VideoCapture(animation_path)
        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            arr_images.append(frame)
        cap.release()
        return arr_images

    def get_pos_pil(self, pos: int) -> Image.Image:
        return self.pil_images[
            math.floor(
                math.modf((pos - self.start_pos) / self.period)[0]
                * self.pil_frame_count
            )
        ]

    def get_pos_arr(self, pos: int) -> npt.NDArray[np.uint8]:
        return self.arr_images[
            math.floor(
                math.modf((pos - self.start_pos) / self.period)[0]
                * self.pil_frame_count
            )
        ]

    def show_all_frames(
        self,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1,
        ncols: int = 6,
        figsize: Optional[Tuple[int, int]] = None,
        fig: Optional[Figure] = None,
    ) -> Figure:
        fig = show_frames(
            video=self.animation_path,
            start=start,
            end=end,
            step=step,
            nframes=self.arr_frame_count,
            ncols=ncols,
            figsize=figsize,
            fig=fig,
        )
        return fig

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
