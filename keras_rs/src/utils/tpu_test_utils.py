import contextlib
import os
import threading
from typing import Any, Callable, ContextManager, Optional, Tuple, Union

import keras
import tensorflow as tf


class DummyStrategy:
    def scope(self) -> ContextManager[None]:
        return contextlib.nullcontext()

    @property
    def num_replicas_in_sync(self) -> int:
        return 1

    def run(self, fn: Callable[..., Any], args: Tuple[Any, ...]) -> Any:
        return fn(*args)

    def experimental_distribute_dataset(
        self, dataset: Any, options: Optional[Any] = None
    ) -> Any:
        del options
        return dataset


class JaxDummyStrategy(DummyStrategy):
    @property
    def num_replicas_in_sync(self) -> Any:
        import jax

        return jax.device_count("tpu")


StrategyType = Union[tf.distribute.Strategy, DummyStrategy]

_shared_strategy: Optional[StrategyType] = None
_lock = threading.Lock()


def create_tpu_strategy() -> Optional[StrategyType]:
    """Initializes the TPU system and returns a TPUStrategy."""
    print("Attempting to create TPUStrategy...")
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="")
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print(
            "TPUStrategy created successfully."
            "Devices: {strategy.extended.num_replicas_in_sync}"
        )
        return strategy
    except Exception as e:
        print(f"Error creating TPUStrategy: {e}")
        return None


def get_shared_tpu_strategy() -> Optional[StrategyType]:
    """
    Returns a session-wide shared TPUStrategy instance.
    Creates the instance on the first call.
    Returns None if not in a TPU environment or if creation fails.
    """
    global _shared_strategy
    if _shared_strategy is not None:
        return _shared_strategy

    with _lock:
        if _shared_strategy is None:
            if "TPU_NAME" not in os.environ:
                _shared_strategy = DummyStrategy()
                return _shared_strategy
            if keras.backend.backend() == "tensorflow":
                resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
                tf.config.experimental_connect_to_cluster(resolver)
                topology = tf.tpu.experimental.initialize_tpu_system(resolver)
                tpu_metadata = resolver.get_tpu_system_metadata()
                device_assignment = tf.tpu.experimental.DeviceAssignment.build(
                    topology, num_replicas=tpu_metadata.num_hosts
                )
                _shared_strategy = tf.distribute.TPUStrategy(
                    resolver, experimental_device_assignment=device_assignment
                )
                print("### num_replicas", _shared_strategy.num_replicas_in_sync)
            elif keras.backend.backend() == "jax":
                _shared_strategy = JaxDummyStrategy()
                print("### num_replicas", _shared_strategy.num_replicas_in_sync)
            else:
                _shared_strategy = DummyStrategy()
    return _shared_strategy


def run_with_strategy(
    strategy: Any,
    fn: Callable[..., Any],
    *args: Any,
    jit_compile: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Final wrapper fix: Flattens allowed kwargs into positional args before
    entering tf.function to guarantee a fixed graph signature.
    """
    if keras.backend.backend() == "tensorflow":
        all_inputs = (args, kwargs)

        @tf.function(jit_compile=jit_compile)  # type: ignore[untyped-decorator]
        def tf_function_wrapper(input_tuple: Tuple[Any, Any]) -> Any:
            core_args, core_kwargs = input_tuple
            if core_kwargs:
                return strategy.run(fn, args=core_args, kwargs=core_kwargs)
            else:
                return strategy.run(fn, args=core_args)

        return tf_function_wrapper(all_inputs)
    else:
        assert not jit_compile
        return fn(*args, **kwargs)
