# coding: utf-8
import os
from abc import ABC
from typing import Dict, Optional

from ..utils._colorings import toBLUE, toGREEN
from ..utils._loggers import get_logger


class BaseElement(ABC):
    ELEMENT_IDX: int = 0

    def __init__(self):
        self.logger = get_logger(name=self.element_name)
        BaseElement.ELEMENT_IDX += 1

    @property
    def element_name(self):
        return f"{BaseElement.ELEMENT_IDX}.{self.__class__.__name__}"

    def set_attribute(self, name: str, value: str, msg: Optional[str] = None) -> None:
        """Set attribute to this class with logs using ``setattr``.

        Args:
            name (str)          : An attribute name.
            value (str)         : An attribute value.
            msg (str, optional) : Additional log message. Defaults to ``""``.

        Examples:
            >>> from veditor.chaptors import MarqueeEditor
            >>> editor = MarqueeEditor
            >>> editor.set_attribute(name="hoge", value=1)
            >>> hasattr(editor, "hoge")
            True
            >>> editor.hoge
            1
        """
        if msg is None:
            msg = ""
        else:
            msg = str(msg)
            if len(msg) == 0:
                msg = str(value)
            msg = " " + msg
        self.logger.info(f"Set attribute {toGREEN(name)}.{msg}")
        setattr(self, name, value)
