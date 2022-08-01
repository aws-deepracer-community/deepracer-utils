from . import logs, tracks, boto3_enhancer, console

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
