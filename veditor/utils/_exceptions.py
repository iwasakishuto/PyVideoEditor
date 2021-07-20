# coding: utf-8
__all__ = ["PyVideoEditorImprementationError"]


class PyVideoEditorImprementationError(Exception):
    """
    Warnings that developers will resolve.
    Developers are now solving in a simple stupid way.
    """


class KeyError(KeyError):
    def __str__(self):
        return ", ".join(self.args)
