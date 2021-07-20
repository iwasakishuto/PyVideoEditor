# coding: utf-8
import os
import sys
import time
import urllib
from typing import Callable, Optional

from ._colorings import toBLUE, toGREEN, toRED
from .generic_utils import readable_bytes

# Use Specific Opener
opener = urllib.request.build_opener()
opener.addheaders = [
    (
        "User-Agent",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36",
    )
]
urllib.request.install_opener(opener)


def progress_reporthook_create(
    filename: str = "",
    bar_width: int = 20,
    verbose: bool = True,
) -> Callable[[int, int, int], None]:
    """Create Progress reporthook for ``urllib.request.urlretrieve``

    Args:
        filename (str, optional)  : A downloading filename. Defaults to ``""``.
        bar_width (int, optional) : The maximum width of progress bar.. Defaults to ``20``.
        verbose (bool, optional)  : Whether to show progress or not.

    Returns:
        Callable[[int,int,int],None]: The ``reporthook`` which is a callable that accepts a ``block number``, a ``read size``, and the ``total file size`` of the URL target.

    Examples:
        >>> import urllib
        >>> from veditor.utils import progress_reporthook_create
        >>> urllib.request.urlretrieve(url="hoge.zip", filename="hoge.zip", reporthook=progress_reporthook_create(filename="hoge.zip"))
        hoge.zip	1.5%[--------------------] 21.5[s] 8.0[GB/s]	eta 1415.1[s]
    """

    def progress_reporthook_verbose(block_count: int, block_size: int, total_size: int):
        global _reporthook_start_time
        if block_count == 0:
            _reporthook_start_time = time.time()
            return
        progress_size = block_count * block_size
        percentage = min(1.0, progress_size / total_size)
        progress_bar = ("#" * int(percentage * bar_width)).ljust(bar_width, "-")

        duration = time.time() - _reporthook_start_time
        speed = progress_size / duration
        eta = (total_size - progress_size) / speed

        speed, speed_unit = readable_bytes(speed)

        sys.stdout.write(
            f"\r{filename}\t{percentage:.1%}[{progress_bar}] {duration:.1f}[s] {speed:.1f}[{speed_unit}/s]\teta {eta:.1f}[s]"
        )
        if progress_size >= total_size:
            print()

    def progress_reporthook_non_verbose(block_count, block_size, total_size):
        pass

    return progress_reporthook_verbose if verbose else progress_reporthook_non_verbose


def download_file(
    url: str,
    dirname: str = ".",
    filename: Optional[str] = None,
    bar_width: int = 20,
    verbose: bool = True,
) -> str:
    """Download a file from ``url``

    Args:
        url (str)                          : URL where the data is located.
        dirname (str, optional)            : The directory where downloaded data will be saved.. Defaults to ``"."``.
        filename (Optional[str], optional) : The name of the file you want to download. Saved with this name. Defaults to ``None``.
        bar_width (int, optional)          : The maximum width of progress bar. Defaults to ``20``.
        verbose (bool, optional)           : Whether to show progress or not. Defaults to ``True``.

    Returns:
        str: The path to a downloaded file.

    Examples:
        >>> from veditor.utils import download_file
        >>> download_file(url="https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml")
        Download a file from https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
                    * Content-Encoding : None
                    * Content-Length   : (333.404296875, 'MB')
                    * Content-Type     : text/plain; charset=utf-8
                    * Save Destination : ./haarcascade_eye.xml
        haarcascade_eye.xml	100.0%[####################] 0.1[s] 5.5[GB/s]	eta -0.0[s]
        './haarcascade_eye.xml'
    """

    try:
        with urllib.request.urlopen(url) as web_file:
            # Get Information from webfile header
            headers = dict(web_file.headers._headers)
        content_encoding = headers.get("Content-Encoding")
        content_length, unit = readable_bytes(int(headers.get("Content-Length", 0)))
        content_length = f"{content_length:.1f} [{unit}]"
        content_type = headers.get("Content-Type")
        filename = filename or url.split("/")[-1]
        path = os.path.join(dirname, filename)
        if verbose:
            print(
                f"""Download a file from {toBLUE(url)}
    * Content-Encoding : {toGREEN(content_encoding)}
    * Content-Length   : {toGREEN(content_length)}
    * Content-Type     : {toGREEN(content_type)}
    * Save Destination : {toBLUE(path)}"""
            )
        _, res = urllib.request.urlretrieve(
            url=url,
            filename=path,
            reporthook=progress_reporthook_create(
                filename=filename, bar_width=bar_width, verbose=verbose
            ),
        )
    except urllib.error.URLError as e:
        print(f"{toRED(e)} : url={toBLUE(url)}")
    return path
