"""Utilities for lazily exposing models without forcing heavy imports."""

from importlib import import_module

__all__ = [
    'PatchAutoencoder',
    'SimpleTransformer',
    'preprocess_data',
    'load_model'
]


def __getattr__(name):
    if name in {'PatchAutoencoder', 'SimpleTransformer'}:
        module = import_module('.transformer_model', __name__)
    elif name in {'preprocess_data', 'load_model'}:
        module = import_module('.utils', __name__)
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    return getattr(module, name)
