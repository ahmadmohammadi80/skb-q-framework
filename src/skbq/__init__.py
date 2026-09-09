"""SKB-Q research framework package."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("skb-q-framework")
except PackageNotFoundError:
    __version__ = "0.1.0"


__all__ = ["__version__"]
