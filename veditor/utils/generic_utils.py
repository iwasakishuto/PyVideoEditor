# coding: utf-8
import argparse
import datetime
import re
from typing import Any, List

from ._colorings import toBLUE, toGREEN, toRED
from ._exceptions import KeyError

NoneType = type(None)


def handleKeyError(lst: List[Any], **kwargs):
    """Check whether all ``kwargs.values()`` in the ``lst``.

    Args:
        lst (List[Any])   : candidates.
        kwargs (dict)     : ``key`` is the varname that is easy to understand when an error occurs

    Examples:
        >>> from veditor.utils import handleKeyError
        >>> handleKeyError(lst=range(3), val=1)
        >>> handleKeyError(lst=range(3), val=100)
        KeyError: Please choose the argment val from ['0', '1', '2']. you chose 100
        >>> handleKeyError(lst=range(3), val1=1, val2=2)
        >>> handleKeyError(lst=range(3), val1=1, val2=100)
        KeyError: Please choose the argment val2 from ['0', '1', '2']. you chose 100

    Raise:
        KeyError: If ``kwargs.values()`` not in the ``lst``
    """
    for k, v in kwargs.items():
        if v not in lst:
            lst = ", ".join([f"'{toGREEN(e)}'" for e in lst])
            raise KeyError(
                f"Please choose the argment {toBLUE(k)} from [{lst}]. you chose {toRED(v)}"
            )


def class2str(class_: object) -> str:
    """Convert class to str.

    Args:
        class_ (object): class object

    Returns:
        str : Class name.

    Examples:
        >>> from veditor.utils import class2str
        >>> class2str(str)
        'str'
        >>> class2str(tuple)
        'tuple'
    """
    return re.sub(r"<class '(.*?)'>", r"\1", str(class_))


def handleTypeError(types: List = [Any], **kwargs):
    """Check whether all types of ``kwargs.values()`` match any of ``types``.

    Args:
        types (List[Any]) : Candidate types.
        kwargs (dict)     : ``key`` is the varname that is easy to understand when an error occurs

    Examples:
        >>> from veditor.utils import handleTypeError
        >>> handleTypeError(types=[str], val="foo")
        >>> handleTypeError(types=[str, int], val=1)
        >>> handleTypeError(types=[str, int], val=1.)
        TypeError: val must be one of ['str', 'int'], not float
        >>> handleTypeError(types=[str], val1="foo", val2="bar")
        >>> handleTypeError(types=[str, int], val1="foo", val2=1.)
        TypeError: val2 must be one of ['str', 'int'], not float

    Raise:
        TypeError: If the types of ``kwargs.values()`` are none of the ``types``
    """
    types = tuple([NoneType if e is None else e for e in types])
    for k, v in kwargs.items():
        if not isinstance(v, types):
            str_true_types = ", ".join([f"'{toGREEN(class2str(t))}'" for t in types])
            srt_false_type = class2str(type(v))
            if len(types) == 1:
                err_msg = f"must be {str_true_types}"
            else:
                err_msg = f"must be one of [{str_true_types}]"
            raise TypeError(f"{toBLUE(k)} {err_msg}, not {toRED(srt_false_type)}")


def str_strip(string: str) -> str:
    """Convert all consecutive whitespace  characters to `' '` (half-width whitespace), then return a copy of the string with leading and trailing whitespace removed.

    Args:
        string (str) : string

    Returns:
        str : A copy of the string with leading and trailing whitespace removed

    Example:
        >>> from veditor.utils import str_strip
        >>> str_strip(" hoge   ")
        'hoge'
        >>> str_strip(" ho    ge   ")
        'ho ge'
        >>> str_strip("  ho    g　e")
        'ho g e'
    """
    return re.sub(pattern=r"[\s 　]+", repl=" ", string=str(string)).strip()


def now_str(tz=None, fmt="%Y-%m-%d@%H.%M.%S"):
    """Returns new datetime string representing current time local to ``tz`` under the control of an explicit format string.

    Args:
        tz (datetime.timezone) : Timezone object. If no ``tz`` is specified, uses local timezone.
        fmt (str)              : format string. See `Python Documentation <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_

    Returns:
        str : A datetime string.

    Example:
        >>> from veditor.utils import now_str
        >>> now_str()
        '2020-09-14@22.31.17'
        >>>now_str(fmt="%A, %d. %B %Y %I:%M%p")
        Monday, 14. September 2020 10:31PM'
        >>> now_str(tz=datetime.timezone.utc)
        '2020-09-14@13.31.17'
    """
    return datetime.datetime.now(tz=tz).strftime(fmt)


def flatten_dual(lst: List[List[Any]]) -> List[Any]:
    """Flatten double list.

    Args:
        lst (List[List[Any]]): Dual list.

    Returns:
        List[Any] : Flattened single list.

    Example:
        >>> from pycharmers.utils import flatten_dual
        >>> flatten_dual([[1,2,3],[4,5,6]])
        [1, 2, 3, 4, 5, 6]
        >>> flatten_dual([[[1,2,3]],[4,5,6]])
        [[1, 2, 3], 4, 5, 6]
        >>> flatten_dual(flatten_dual([[[1,2,3]],[4,5,6]]))
        TypeError: 'int' object is not iterable

    Raise:
        TypeError: If list is not a dual list.
    """
    return [element for sublist in lst for element in sublist]
