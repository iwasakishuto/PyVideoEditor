# coding: utf-8
import argparse
import datetime
import re
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union

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


def readable_bytes(byte: Number) -> Tuple[Number, str]:
    """Unit conversion for readability.

    Args:
        byte (Number): File byte [B].

    Examples:
        >>> from veditor.utils import readable_bytes
        >>> for i in range(1,30,3):
        ...     byte = pow(10,i)
        ...     size, unit = readable_bytes(pow(10,i))
        ...     print(f"{byte:.1g}[B] = {size:.2f}[{unit}]")
        1e+01[B] = 10.00[B]
        1e+04[B] = 9.77[KB]
        1e+07[B] = 9.54[MB]
        1e+10[B] = 9.31[GB]
        1e+13[B] = 9.09[TB]
        1e+16[B] = 8.88[PB]
        1e+19[B] = 8.67[EB]
        1e+22[B] = 8.47[ZB]
        1e+25[B] = 8.27[YB]
        1e+28[B] = 8271.81[YB]
    """
    units = ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]
    for unit in units:
        if (abs(byte) < 1024.0) or (unit == units[-1]):
            break
        byte /= 1024.0  # size >> 10
    return (byte, unit + "B")


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


_trbl: List[str] = ["top", "right", "bottom", "left"]


def assign_trbl(
    data: Dict[str, Any],
    name: str,
    default: Union[Number, List[Number]] = 0,
    ret_name: bool = False,
) -> Union[
    Tuple[Tuple[Number, Number, Number, Number], Tuple[str, str, str, str]],
    Tuple[Number, Number, Number, Number],
]:
    """Return the ``name`` 's values of [``Top``, ``Right``, ``Bottom``, ``Left``] from ``data``. Determine the each position as well as css.

    Args:
        data (Dict[str,Any])                           : A dictionary which stores data.
        name (str)                                     : The name of the value you want to assign..
        default (Union[Number,List[Number]], optional) : Default Value. Defaults to ``0``.
        ret_name (bool, optional)                      : Whether to return names or not. Defaults to ``False``.

    Returns:
        Union[Tuple[Tuple[Number, Number, Number, Number], Tuple[str,str,str,str]], Tuple[Number, Number, Number, Number]]: Values of ``Top``, ``Right``, ``Bottom``, ``Left``. If ``ret_name`` is ``True``, add names.

    Examples:
        >>> from veditor.utils import assign_trbl
        >>> assign_trbl(data={"margin": [1,2,3,4]}, name="margin")
        (1, 2, 3, 4)
        >>> assign_trbl(data={"margin": [1,2,3]}, name="margin")
        (1, 2, 3, 2)
        >>> assign_trbl(data={"margin": [1,2]}, name="margin")
        (1, 2, 1, 2)
        >>> assign_trbl(data={"margin": 1}, name="margin")
        (1, 1, 1, 1)
        >>> assign_trbl(data={"margin": 1}, name="padding", default=5)
        (5, 5, 5, 5)
    """
    vals = data.get(name, default)
    if isinstance(vals, (Number, NoneType)):
        vals = [vals]

    if len(vals) == 0:
        t = r = b = l = None
    elif len(vals) == 1:
        t = r = b = l = vals[0]
    elif len(vals) == 2:
        t = b = vals[0]
        l = r = vals[1]
    elif len(vals) == 3:
        t, r, b = vals
        l = r
    elif len(vals) >= 4:
        t, r, b, l = vals[:4]

    ret: List[Number] = []
    names: List[str] = []
    for s, v in zip(_trbl, [t, r, b, l]):
        _name = f"{name}_{s}"
        ret.append(data.get(_name.replace("_", "-"), data.get(_name, v)))
        names.append(_name)

    if ret_name:
        return (tuple(ret), tuple(names))

    return tuple(ret)
