# coding: utf-8
import argparse
import sys

import cv2
import numpy as np

from veditor.cli import cvui

from ..__meta__ import __package_name__

WINDOW_NAME = f"Check Clip Location {__package_name__}"


def check_clip_locations(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        prog="check-clip-locations",
        description="Find out where to clip the video.",
        add_help=True,
    )
    parser.add_argument("video", type=str, help="Path to the input video file.")
    parser.add_argument("--UI-width", type=int, default=300)
    args = parser.parse_args(argv)
    cap = cv2.VideoCapture(args.video)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ui_width = args.UI_width

    topValue = [0]
    leftValue = [0]
    rightValue = [width]
    bottomValue = [height]
    posValue = [0]

    curt_pos = 0
    bg = np.zeros(shape=(height, width + ui_width, 3), dtype=np.uint8)
    cvui.init(WINDOW_NAME)
    while True:
        bg[:] = (49, 52, 49)

        # TrackBar for Video Positions
        cvui.trackbar(
            bg,
            width + 20,
            50,
            ui_width - 100,
            posValue,
            0,
            count,
            options=cvui.TRACKBAR_DISCRETE,
            discreteStep=max(1, count // 10),
        )

        for i, (name, Value) in enumerate(
            zip(
                ["top", "left", "right", "bottom"],
                [topValue, leftValue, rightValue, bottomValue],
            )
        ):
            cvui.counter(bg, width + 20, 200 + 80 * i, Value, 1)
            cvui.text(bg, width + 20, 240 + 80 * i, f"Current {name} value: {Value[0]}")

        cvui.update()

        cap.set(cv2.CAP_PROP_POS_FRAMES, posValue[0])
        ret, frame = cap.read()
        if frame is not None:
            t = max(0, topValue[0])
            l = max(0, leftValue[0])
            r = min(width, rightValue[0])
            b = min(height, bottomValue[0])

            w = r - l
            h = b - t
            t_ = (height - h) // 2
            l_ = (width - w) // 2
            bg[t_ : t_ + h, l_ : l_ + w, :] = frame[t:b, l:r]

        # Show everything on the screen
        cv2.imshow(WINDOW_NAME, bg)

        # Check if ESC key was pressed
        if cv2.waitKey(20) == cvui.ESCAPE:
            break
    cv2.destroyWindow(WINDOW_NAME)
    cap.release()
