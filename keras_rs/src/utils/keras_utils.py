from typing import Union

import keras

from keras_rs.src import types


def clone_initializer(
    initializer: Union[str, keras.initializers.Initializer],
) -> keras.initializers.Initializer:
    """Clones an initializer to ensure a new seed.

    Args:
        initializer: The initializer to clone.

    Returns:
        A cloned initializer if it is clonable, otherwise the original one.

    As of tensorflow 2.10, we need to clone user passed initializers when
    invoking them twice to avoid creating the same randomized initialization.
    """
    if isinstance(initializer, keras.initializers.Initializer):
        config = initializer.get_config()
        initializer_class: type[keras.initializers.Initializer] = (
            initializer.__class__
        )
        return initializer_class.from_config(config)
    # If we get a string or dict, just return as we cannot and should not clone.
    return initializer


def check_shapes_compatible(
    shape1: types.TensorShape, shape2: types.TensorShape
) -> bool:
    # Check rank first.
    if len(shape1) != len(shape2):
        return False

    for d1, d2 in zip(shape1, shape2):
        if isinstance(d1, int) and isinstance(d2, int):
            if d1 != d2:
                return False

    return True
