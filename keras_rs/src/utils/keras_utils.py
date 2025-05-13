from typing import Any, Callable

import keras

from keras_rs.src import types


def no_automatic_dependency_tracking(
    fn: Callable[..., Any],
) -> Callable[..., Any]:
    """Decorator to disable automatic dependency tracking in Keras and TF.

    Args:
        fn: the function to disable automatic dependency tracking for.

    Returns:
        a wrapped version of `fn`.
    """
    if keras.backend.backend() == "tensorflow":
        import tensorflow as tf

        fn = tf.__internal__.tracking.no_automatic_dependency_tracking(fn)

    wrapped_fn: Callable[..., Any] = (
        keras.src.utils.tracking.no_automatic_dependency_tracking(fn)
    )
    return wrapped_fn


def clone_initializer(
    initializer: types.InitializerLike,
) -> types.InitializerLike:
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


def check_shapes_compatible(shape1: types.Shape, shape2: types.Shape) -> bool:
    # Check rank first.
    if len(shape1) != len(shape2):
        return False

    for d1, d2 in zip(shape1, shape2):
        if isinstance(d1, int) and isinstance(d2, int):
            if d1 != d2:
                return False

    return True


def check_rank(
    x_rank: int,
    allowed_ranks: tuple[int, ...],
    tensor_name: str,
) -> None:
    if x_rank not in allowed_ranks:
        raise ValueError(
            f"`{tensor_name}` should have a rank from `{allowed_ranks}`."
            f"Received: `{x_rank}`."
        )
