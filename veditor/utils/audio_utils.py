# coding: utf-8
import logging
import os
import subprocess
from typing import List, Optional

from pydub import AudioSegment

from ._colorings import toBLUE
from .generic_utils import openf


def synthesize_audio(
    video_path: str,
    audio_path: str,
    out_path: Optional[str] = None,
    start: int = 0,
    end: int = -1,
    volume: int = 0,
    offset: str = "00:00:00",
    open: bool = True,
    delete_intermidiates: bool = False,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Use ``ffmpeg`` directly or ``moviepy`` to synthesize audio (at ``audio_path``) to video (at ``video_path``)

    Args:
        video_path (str)                   : The path to video fiile.
        audio_path (str)                   : The path to audio (video) fiile.
        out_path (Optional[str], optional) : The path to the created video (with audio) file. Defaults to ``None``.
        offset (str, optional)             : Offset until the voice starts. Defaults to ``"00:00:00"``.
        open (bool, optional)              : Whether to open the created video or not. Defaults to ``True``.
        delete_silence (bool, optional)    : Whether to delete the silence video (``video_path``) or not. Defaults to ``False``.

    Returns:
        str: The path to the created video (with audio) file.

    Examples:
        >>> from veditor.utils import synthesize_audio
        >>> # Prepare Audio file (.mp3)
        >>> synthesize_audio(audio_path="sound.mp3", video_path="no_sound.mp4")
        >>> # Prepare Video with Audio file (.mp4)
        >>> synthesize_audio(audio_path="sound.mp4", video_path="no_sound.mp4")
    """
    root, ext = os.path.splitext(audio_path)
    intermediate_files: List[str] = []
    if ext not in [".mp3", ".wav"]:
        audio = AudioSegment.from_file(file=audio_path, format=ext[1:])
        ext = ".mp3"
        audio_path = root + ext
        audio.export(out_f=root + ext, format=ext[1:])
        intermediate_files.append(audio_path)
    if start != 0 or end != -1:
        audio = AudioSegment.from_file(file=audio_path, format=ext[1:])
        audio_path = f"{root}_{start}-{end}-{volume}{ext}"
        (audio[start:end] + volume).export(out_f=audio_path, format="mp3")
        intermediate_files.append(audio_path)
    if out_path is None:
        out_path = f"_synthesized".join(os.path.splitext(video_path))
    # Append Audio.
    command = f"ffmpeg -y -itsoffset {offset} -i '{video_path}' -i '{audio_path}' -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 '{out_path}' -async 1 -strict -2"
    if logger is not None:
        logger.info(f"Run the following command:\n{command}")
    subprocess.call(command, shell=True)
    # Open the created video.
    if open:
        openf(out_path)
    # Delete the silence video.
    if delete_intermidiates:
        for fp in intermediate_files:
            os.remove(fp)
    return out_path


def overlay_audio(
    base_media_path: str,
    overlay_media_path: str,
    out_path: Optional[str] = None,
    position: int = 0,
) -> str:
    """Overlay audio at ``overlay_media_path`` on audio at ``base_media_path``.

    Args:
        base_media_path (str)              : The path to media file (contains audio) to be overlayed.
        overlay_media_path (str)           : The path to media file (contains audio) to overlay.
        out_path (Optional[str], optional) : Path to the created audio file. Defaults to ``None``.
        position (int, optional)           : The position (``[ms]``) to start overlaying the provided segment in to this one. Defaults to ``0``.

    Returns:
        str: Path to the created audio file.
    """
    root, ext = os.path.splitext(base_media_path)
    base_audio = AudioSegment.from_file(file=base_media_path, format=ext[1:])
    overlay_audio = AudioSegment.from_file(file=overlay_media_path)
    if out_path is None:
        out_path = f"{root}_overlayed.mp3"
    base_audio.overlay(overlay_audio, position=int(position)).export(out_path)
    return out_path
