# coding: utf-8

from .base import BaseElement


class TextElement(BaseElement):
    def __init__(self, text: str, ttfontname: str):
        super().__init__()
        self.ttfontname = ttfontname
