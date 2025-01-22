import os

# Import everything from /api/ into keras_rs.
from keras_rs.api import *  # noqa: F403

# Import * ignores names starting with "_", and `__version__` comes from
# `version` anyway.
from keras_rs.src.version import __version__

# Add everything in /api/ to the module search path.
__path__.append(os.path.join(os.path.dirname(__file__), "api"))  # noqa: F405

# Don't pollute namespace.
del os


# Never autocomplete `.src` or `.api` on an imported keras_rs object.
def __dir__() -> list[str]:
    keys = dict.fromkeys((globals().keys()))
    keys.pop("src")
    keys.pop("api")
    return list(keys)


# Don't import `.src` or `.api` during `from keras_rs import *`.
__all__ = [
    name
    for name in globals().keys()
    if not (name.startswith("_") or name in ("src", "api"))
]
