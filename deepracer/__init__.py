from . import logs, tracks

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("deepracer-utils")
    except PackageNotFoundError:
        __version__ = "unknown"
except ImportError:
    __version__ = "unknown"
