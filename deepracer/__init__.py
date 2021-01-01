import os

DEEPRACER_UTILS_ROOT = os.path.dirname(os.path.abspath(__file__))

from . import logs, tracks, boto3_enhancer

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
