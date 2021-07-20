# coding: utf-8
import logging
import logging.config
from typing import Optional

from ..utils._colorings import toACCENT

__all__ = ["get_logger"]

_loggers = {}


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger with the specified name, creating it if necessary.

    Args:
        name (Optional[str], optional) : The logger name. If no ``name`` is specified, return the root logger. Defaults to ``None``.

    Returns:
        logging.Logger: An instance of ``logging.Logger``.
    """
    global _loggers
    if name in _loggers:
        return _loggers[name]
    logger = logging.getLogger(name)
    streamhandler = logging.StreamHandler()
    formatter = logging.Formatter(
        f"[{toACCENT(name)}] %(asctime)s [%(levelname)s]: %(message)s"
    )
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)
    logger.setLevel(logging.DEBUG)
    _loggers[name] = logger
    return logger
