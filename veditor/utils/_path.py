# coding: utf-8
import os
from pathlib import Path

from ._colorings import toBLUE

__all__ = [
    "UTILS_DIR",
    "MODULE_DIR",
    "VEDITOR_DIR",
]


def _makedirs(name: str, mode: int = 0o777, msg: str = "", verbose: bool = True):
    """Create a directory if it does not exist.

    Args:
        name (str)               : Path to the directory you want to create.
        mode (int, optional)     : Permissions or Special modes of a directory. Defaults to ``0o777``.
        msg (str, optional)      : Additional information to show. Defaults to ``""``.
        verbose (bool, optional) : Whether to show a message or not. Defaults to ``True``.
    """
    if not os.path.exists(name):
        os.makedirs(name=name, mode=mode)
        if verbose:
            print(f"{toBLUE(name)} is created.{msg}")


UTILS_DIR = os.path.dirname(os.path.abspath(__file__))  # path/to/veditor/utils
MODULE_DIR = os.path.dirname(UTILS_DIR)  # path/to/veditor
VEDITOR_DIR = os.path.join(
    os.path.expanduser("~"), ".veditor"
)  # /Users/<username>/.veditor
# Check whether uid/gid has the write access to VEDITOR_DIR
if os.path.exists(VEDITOR_DIR) and (not os.access(VEDITOR_DIR, os.W_OK)):
    VEDITOR_DIR = os.path.join("/tmp", ".veditor")
_makedirs(name=VEDITOR_DIR)

SAMPLE_IMAGE_PATH = os.path.join(VEDITOR_DIR, "file_example_MP4_1280_10MG.mp4")
SAMPLE_VIDEO_PATH = os.path.join(VEDITOR_DIR, "file_example_PNG_500kB.png")
SAMPLE_FONT_PATH = os.path.join(VEDITOR_DIR, "file_example_MP4_1280_10MG.mp4")
