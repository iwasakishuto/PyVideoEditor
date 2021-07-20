# coding: utf-8
import os

from ._path import FONT_DIR, VEDITOR_DIR
from .download_utils import download_file


class SampleData:

    DATASETS = {
        "IMAGE_PATH": {
            "path": os.path.join(VEDITOR_DIR, "file_example_PNG_500kB.png"),
            "url": "https://file-examples-com.github.io/uploads/2017/10/file_example_PNG_500kB.png",
        },
        "VIDEO_PATH": {
            "path": os.path.join(VEDITOR_DIR, "file_example_MP4_1280_10MG.mp4"),
            "url": "https://file-examples-com.github.io/uploads/2017/04/file_example_MP4_1280_10MG.mp4",
        },
        "AUDIO_PATH": {
            "path": os.path.join(VEDITOR_DIR, "file_example_MP3_700KB.mp3"),
            "url": "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_700KB.mp3",
        },
        "FONT_POKEFONT_PATH": {
            "path": os.path.join(FONT_DIR, "pokefont", "pokemon-font.ttf"),
            "url": "https://github.com/PascalPixel/pokemon-font/blob/60280120447da9de4f0f28ceaacff144642bb16a/fonts/pokemon-font.ttf?raw=true",
        },
    }

    def __init__(self):
        for name, info in SampleData.DATASETS.items():
            path = info["path"]
            url = info["url"]
            head, tail = os.path.split(path)
            if not os.path.exists(path):
                download_file(url=url, dirname=head, filename=tail, verbose=True)
            setattr(self, name, path)
