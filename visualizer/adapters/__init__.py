"""Model adapter interface and registry."""

from .base import GeneratorAdapter
from .hooks import HookMixin
from .registry import get_adapter, list_adapters, register_adapter, discover_adapters
from .dmd2_imagenet import DMD2ImageNetAdapter

__all__ = [
    "GeneratorAdapter",
    "HookMixin",
    "get_adapter",
    "list_adapters",
    "register_adapter",
    "discover_adapters",
    "DMD2ImageNetAdapter",
]
