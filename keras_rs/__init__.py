# This file should NEVER be packaged! This is a hack to make "import keras_rs"
# from the base of the repo import the api correctly. We'll keep it for compat.

import os

# Add everything in /api/ to the module search path.
__path__.append(os.path.join(os.path.dirname(__file__), "api"))  # noqa: F405

# Import everything from /api/ into keras_rs.
from keras_rs.api import *  # noqa: F403, E402

# Import * ignores names starting with "_", and `__version__` comes from
# `version` anyway.
from keras_rs.src.version import __version__  # noqa: E402

# Don't pollute namespace.
del os
