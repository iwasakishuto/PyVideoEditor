# coding: utf-8
import argparse
import sys

from ..utils.argparse_utils import DictParamProcessor


def func(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(prog="command-name", add_help=True)
    parser.add_argument(
        "-P",
        "--params",
        default={},
        action=DictParamProcessor,
        help="Specify the kwargs. You can specify by -P username=USERNAME -P password=PASSWORD",
    )
    args = parser.parse_args(argv)
