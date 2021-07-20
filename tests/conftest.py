# coding: utf-8
import os
import sys
import pytest
from data import TestData
import warnings
try:
    from veditor.utils._warnings import PyVideoEditorImprementationWarning
except ModuleNotFoundError:
    here     = os.path.abspath(os.path.dirname(__file__))
    REPO_DIR = os.path.dirname(here)
    sys.path.append(REPO_DIR)
    print(f"You didn't install 'PyVideoEditor', so add {REPO_DIR} to search path for modules.")
    from lib.utils._warnings import PyVideoEditorImprementationWarning

def pytest_addoption(parser):
    parser.addoption("--veditor-warnings", choices=["error", "ignore", "always", "default", "module", "once"], default="ignore")

def pytest_configure(config):
    action = config.getoption("veditor_warnings")
    warnings.simplefilter(action, category=PyVideoEditorImprementationWarning)

@pytest.fixture
def db():
    database = TestData()
    return database