import contextlib
import os

import keras
import tensorflow as tf

try:
    import jax
except ImportError:
    jax = None


class DummyStrategy:
    def scope(self):
        return contextlib.nullcontext()

    @property
    def num_replicas_in_sync(self):
        return 1

    def run(self, fn, args):
        return fn(*args)

    def experimental_distribute_dataset(self, dataset, options=None):
        del options
        return dataset


class JaxDummyStrategy(DummyStrategy):
    @property
    def num_replicas_in_sync(self):
        if jax is None:
            return 0
        return jax.device_count("tpu")


def get_tpu_strategy(test_case):
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


def run_with_strategy(strategy, fn, *args, jit_compile=False, **kwargs):
    """Wrapper for running a function under a strategy."""
    if keras.backend.backend() == "tensorflow":
        @tf.function(jit_compile=jit_compile)
        def tf_function_wrapper(*tf_function_args):
            return strategy.run(fn, args=tf_function_args, kwargs=kwargs)
        return tf_function_wrapper(*args)
    else:
        assert not jit_compile
        return fn(*args, **kwargs)
