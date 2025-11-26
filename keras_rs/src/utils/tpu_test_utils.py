import contextlib
import os
import threading
from types import ModuleType
from typing import Any, Callable, ContextManager, Optional, Tuple, Union

import keras
import tensorflow as tf

jax: Optional[ModuleType] = None

try:
    import jax
except ImportError:
    pass


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
        if jax is None:
            return 0
        return jax.device_count("tpu")


StrategyType = Union[tf.distribute.Strategy, DummyStrategy, JaxDummyStrategy]

_shared_strategy: Optional[StrategyType] = None
_lock = threading.Lock()

def create_tpu_strategy() -> Optional[StrategyType]:
    """Initializes the TPU system and returns a TPUStrategy."""
    print("Attempting to create TPUStrategy...")
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print(f"TPUStrategy created successfully. Devices: {strategy.extended.num_replicas_in_sync}")
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
                if jax is None:
                    raise ImportError(
                        "JAX backend requires jax to be installed for TPU."
                    )
                print("### num_replicas", jax.device_count("tpu"))
                _shared_strategy = JaxDummyStrategy()
            else:
                _shared_strategy = DummyStrategy()
            if _shared_strategy is None:
                 print("Failed to create the shared TPUStrategy.")
    return _shared_strategy


def get_tpu_strategy(test_case: Any) -> StrategyType:
    """Get TPU strategy if on TPU, otherwise return DummyStrategy."""
    if "TPU_NAME" not in os.environ:
        return DummyStrategy()
    if keras.backend.backend() == "tensorflow":
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        topology = tf.tpu.experimental.initialize_tpu_system(resolver)
        tpu_metadata = resolver.get_tpu_system_metadata()
        device_assignment = tf.tpu.experimental.DeviceAssignment.build(
            topology, num_replicas=tpu_metadata.num_hosts
        )
        strategy = tf.distribute.TPUStrategy(
            resolver, experimental_device_assignment=device_assignment
        )
        print("### num_replicas", strategy.num_replicas_in_sync)
        test_case.addCleanup(tf.tpu.experimental.shutdown_tpu_system, resolver)
        return strategy
    elif keras.backend.backend() == "jax":
        if jax is None:
            raise ImportError(
                "JAX backend requires jax to be installed for TPU."
            )
        print("### num_replicas", jax.device_count("tpu"))
        return JaxDummyStrategy()
    else:
        return DummyStrategy()


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
        # Extract sample_weight and treat it as an explicit third positional
        # argument. If not present, use a placeholder (None).
        sample_weight_value = kwargs.get("sample_weight", None)
        all_inputs = args + (sample_weight_value,)

        @tf.function(jit_compile=jit_compile)  # type: ignore[misc]
        def tf_function_wrapper(input_tuple: Tuple[Any, ...]) -> Any:
            num_original_args = len(args)
            core_args = input_tuple[:num_original_args]
            sw_value = input_tuple[-1]

            if sw_value is not None:
                all_positional_args = core_args + (sw_value,)
                return strategy.run(fn, args=all_positional_args)
            else:
                return strategy.run(fn, args=core_args)

        return tf_function_wrapper(all_inputs)
    else:
        assert not jit_compile
        return fn(*args, **kwargs)
