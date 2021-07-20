# coding: utf-8
import argparse
import re

from .generic_utils import str_strip


def ListParamProcessorCreate(type=str):
    """Create a ListParamProcessor

    Args:
        type (type) : type of each element in list.

    Returns:
        ListParamProcessor (argparse.Action) : Processor which receives list arguments.

    Examples:
        >>> import argparse
        >>> from veditor.utils import ListParamProcessorCreate
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument("--list_params", action=ListParamProcessorCreate())
        >>> args = parser.parse_args(args=["--list_params", "[あ, い, う]"])
        >>> args.list_params
        ['あ', 'い', 'う']
    """

    class ListParamProcessor(argparse.Action):
        """Receive List arguments.

        Examples:
            >>> import argparse
            >>> from veditor.utils import ListParamProcessor
            >>> parser = argparse.ArgumentParser()
            >>> parser.add_argument("--list_params", action=ListParamProcessor)
            >>> args = parser.parse_args(args=["--list_params", "[あ, い, う]"])
            >>> args.list_params
            ['あ', 'い', 'う']

        Note:
            If you run from the command line, execute as follows::

            $ python app.py --list_params "[あ, い, う]"

        """

        def __call__(self, parser, namespace, values, option_strings=None, **kwargs):
            match = re.match(pattern=r"(?:\[|\()(.+)(?:\]|\))", string=values)
            if match:
                values = [type(str_strip(e)) for e in match.group(1).split(",")]
            else:
                values = [type(values)]
            setattr(namespace, self.dest, values)

    return ListParamProcessor


class DictParamProcessor(argparse.Action):
    """Receive an argument as a dictionary.

    Raises:
        ValueError: You must give one argument for each one keyword.

    Examples:
        >>> import argparse
        >>> from veditor.utils import DictParamProcessor
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument("--dict_params", action=DictParamProcessor)
        >>> args = parser.parse_args(args=["--dict_params", "foo = [a, b, c]", "--dict_params", "bar=d"])
        >>> args.dict_params
        {'foo': ['a', 'b', 'c'], 'bar': 'd'}
        >>> args = parser.parse_args(args=["--dict_params", "foo=a, bar=b"])
        ValueError: too many values to unpack (expected 2)

    Note:
        If you run from the command line, execute as follows::

        $ python app.py --dict_params "foo = [a, b, c]" --dict_params bar=c

    """

    def __call__(self, parser, namespace, values, option_strings=None):
        param_dict = getattr(namespace, self.dest) or {}
        k, v = values.split("=")
        match = re.match(pattern=r"\[(.+)\]", string=str_strip(v))
        if match is not None:
            v = [str_strip(e) for e in match.group(1).split(",")]
        else:
            v = str_strip(v)
        param_dict[str_strip(k)] = v
        setattr(namespace, self.dest, param_dict)


class KwargsParamProcessor(argparse.Action):
    """Set a new argument.

    Examples:
        >>> import argparse
        >>> from veditor.utils import KwargsParamProcessor
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument("--kwargs", action=KwargsParamProcessor)
        >>> args = parser.parse_args(args=["--kwargs", "foo=a", "--kwargs", "bar=b"])
        >>> (args.kwargs, args.foo, args.bar)
        (None, 'a', 'b')

    Note:
        If you run from the command line, execute as follows::

        $ python app.py --kwargs foo=a --kwargs bar=b

    """

    def __call__(self, parser, namespace, values, option_strings=None):
        k, v = values.split("=")
        setattr(namespace, k, v)
