import inspect
import typing
from typing import Callable, Union

import jax
import keras
import numpy as np
import pytest
from absl.testing import absltest
from absl.testing import parameterized
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec

from keras_rs.src.layers.embedding import distributed_embedding_config as config
from keras_rs.src.layers.embedding.jax import config_conversion


@pytest.mark.skipif(
    keras.backend.backend() != "jax",
    reason="Backend specific test",
)
class ConfigConversionTest(parameterized.TestCase):
    def assert_initializers_equal(
        self,
        initializer_a: Union[
            str, keras.initializers.Initializer, jax.nn.initializers.Initializer
        ],
        initializer_b: Union[
            str, keras.initializers.Initializer, jax.nn.initializers.Initializer
        ],
    ):
        """Compare two initializers to ensure they are equivalent."""
        if isinstance(initializer_a, str):
            initializer_a = keras.initializers.get(initializer_a)

        if isinstance(initializer_b, str):
            initializer_b = keras.initializers.get(initializer_b)

        # Unwrap if it is a wrapped keras initializer.
        if isinstance(initializer_a, config_conversion.WrappedKerasInitializer):
            initializer_a = initializer_a.initializer

        if isinstance(initializer_b, config_conversion.WrappedKerasInitializer):
            initializer_b = initializer_b.initializer

        if initializer_a == initializer_b:
            return

        # Check that they are the same class.
        self.assertEqual(type(initializer_a), type(initializer_b))

        if isinstance(
            initializer_a, keras.initializers.Initializer
        ) and isinstance(initializer_b, keras.initializers.Initializer):
            self.assertEqual(
                initializer_a.get_config(), initializer_b.get_config()
            )
            return

        raise ValueError(
            f"Initializers {initializer_a} and {initializer_b} are not equal."
        )

    def assert_optimizers_equal(
        self,
        optimizer_a: Union[
            str, keras.optimizers.Optimizer, embedding_spec.OptimizerSpec
        ],
        optimizer_b: Union[
            str, keras.optimizers.Optimizer, embedding_spec.OptimizerSpec
        ],
    ):
        """Compare two optimizers to ensure they are equivalent."""
        if isinstance(optimizer_a, str):
            optimizer_a = keras.optimizers.get(optimizer_a)
        if isinstance(optimizer_b, str):
            optimizer_b = keras.optimizers.get(optimizer_b)

        # Convert JTE to Keras optimizers.
        if isinstance(optimizer_a, embedding_spec.OptimizerSpec):
            optimizer_a = config_conversion.jte_to_keras_optimizer(optimizer_a)
        if isinstance(optimizer_b, embedding_spec.OptimizerSpec):
            optimizer_b = config_conversion.jte_to_keras_optimizer(optimizer_b)

        if optimizer_a == optimizer_b:
            return

        # Check that they are the same class.
        self.assertEqual(type(optimizer_a), type(optimizer_b))

        if isinstance(optimizer_a, keras.optimizers.Optimizer) and isinstance(
            optimizer_b, keras.optimizers.Optimizer
        ):
            # For Keras optimizers, check their configs.
            self.assertEqual(optimizer_a.get_config(), optimizer_b.get_config())
            return

        raise ValueError(
            f"Initializers {optimizer_a} and {optimizer_b} are not equal."
        )

    def assert_table_configs_equal(
        self,
        table_a: Union[config.TableConfig, embedding_spec.TableSpec],
        table_b: Union[config.TableConfig, embedding_spec.TableSpec],
    ):
        """Compare two table configs to ensure they are equivalent."""
        self.assertEqual(table_a.name, table_b.name)
        self.assertEqual(table_a.vocabulary_size, table_b.vocabulary_size)
        self.assertEqual(table_a.embedding_dim, table_b.embedding_dim)
        self.assertEqual(table_a.combiner, table_b.combiner)
        self.assertEqual(
            table_a.max_ids_per_partition, table_b.max_ids_per_partition
        )
        self.assertEqual(
            table_a.max_unique_ids_per_partition,
            table_b.max_unique_ids_per_partition,
        )
        self.assert_initializers_equal(table_a.initializer, table_b.initializer)
        self.assert_optimizers_equal(table_a.optimizer, table_b.optimizer)

    def get_learning_rate(
        self, learning_rate: Union[float, Callable[..., float]], step: int
    ):
        """Gets the learning rate at the provided iteration step.

        Args:
            learning_rate: The learning rate value or function.  If a Callable,
                must either take no parameters, or the step size as a single
                argument.
            step: The iteration step.

        Returns:
            The learning rate at the provided step.

        Raises:
            ValueError if the learning rate is not supported.
        """
        if callable(learning_rate):
            args = inspect.getfullargspec(learning_rate).args
            # If not a function, then it's an object instance, with `self` as
            # the first argument.
            num_args = (
                len(args)
                if inspect.isfunction(learning_rate)
                else len(args) - 1
            )
            if num_args == 0:
                return learning_rate()
            elif num_args == 1:
                return learning_rate(step)
        elif isinstance(learning_rate, float):
            return learning_rate

        raise ValueError(
            f"Unknown learning rate {learning_rate} of type "
            f"{type(learning_rate)}"
        )

    @parameterized.named_parameters(
        ("constant", 0.01),
        ("lambda", lambda: 0.03),
        ("lambda(step)", lambda step: 0.03 / (step + 1)),
        (
            "LearningRateSchedule",
            keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9
            ),
        ),
    )
    def test_learning_rate_conversion(
        self, learning_rate: Union[float, Callable[..., float]]
    ):
        jte_learning_rate = config_conversion.keras_to_jte_learning_rate(
            learning_rate
        )
        keras_learning_rate = config_conversion.jte_to_keras_learning_rate(
            jte_learning_rate
        )

        for step in range(10):
            value = self.get_learning_rate(learning_rate, step)
            jte_value = self.get_learning_rate(jte_learning_rate, step)
            keras_value = self.get_learning_rate(keras_learning_rate, step)

            self.assertEqual(jte_value, value)
            self.assertEqual(keras_value, value)

        # Round-trip conversion.
        self.assertEqual(keras_learning_rate, learning_rate)

    @parameterized.named_parameters(
        (
            "uniform",
            keras.initializers.RandomUniform(
                minval=-0.05, maxval=0.05, seed=10
            ),
        ),
        ("ones", keras.initializers.Ones()),
        ("zeros", "zeros"),
    )
    def test_initializer_conversion(
        self, initializer: Union[keras.initializers.Initializer, str]
    ):
        jte_initializer = typing.cast(
            config_conversion.WrappedKerasInitializer,
            config_conversion.keras_to_jax_initializer(initializer),
        )
        keras_initializer = config_conversion.jax_to_keras_initializer(
            jte_initializer
        )
        if isinstance(initializer, str):
            initializer = keras.initializers.get(initializer)

        shape = (32, 74)
        value = initializer(shape, dtype=np.float32)
        jte_value = jte_initializer(
            key=jte_initializer.key(), shape=shape, dtype=np.float32
        )
        keras_value = keras_initializer(shape, dtype=np.float32)
        np.testing.assert_array_equal(jte_value, value)
        np.testing.assert_array_equal(keras_value, value)
        self.assert_initializers_equal(keras_initializer, initializer)

    @parameterized.named_parameters(
        # Lazy-load parameters to avoid calling JAX functions before init.
        ("SGD", lambda: keras.optimizers.SGD(learning_rate=0.01)),
        (
            "SGD(ExponentialDecay)",
            lambda: keras.optimizers.SGD(
                learning_rate=keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=1e-2,
                    decay_steps=10000,
                    decay_rate=0.9,
                )
            ),
        ),
        ("Adagrad", lambda: keras.optimizers.Adagrad(learning_rate=0.02)),
        ("string", lambda: "adagrad"),
    )
    def test_optimizer_conversion(
        self,
        lazy_optimizer: Callable[..., Union[keras.optimizers.Optimizer, str]],
    ):
        optimizer = lazy_optimizer()
        jte_optimizer = config_conversion.keras_to_jte_optimizer(optimizer)
        keras_optimizer = config_conversion.jte_to_keras_optimizer(
            jte_optimizer
        )
        if isinstance(optimizer, str):
            optimizer = keras.optimizers.get(optimizer)

        self.assert_optimizers_equal(jte_optimizer, optimizer)
        self.assert_optimizers_equal(keras_optimizer, optimizer)

    def test_table_conversion(self):
        tables = [
            config.TableConfig(
                name="table_a",
                vocabulary_size=32,
                embedding_dim=14,
                initializer="random_normal",
                optimizer="sgd",
                combiner="mean",
                placement="embedding_feature",
                max_ids_per_partition=256,
                max_unique_ids_per_partition=255,
            ),
            config.TableConfig(
                name="table_b",
                vocabulary_size=67,
                embedding_dim=17,
                initializer="random_normal",
                optimizer="adagrad",
                combiner="mean",
                placement="embedding_feature",
                max_ids_per_partition=128,
                max_unique_ids_per_partition=127,
            ),
        ]

        jte_tables = config_conversion.keras_to_jte_table_configs(tables)
        keras_tables = config_conversion.jte_to_keras_table_configs(jte_tables)

        for table, jte_table, keras_table in zip(
            tables, jte_tables, keras_tables
        ):
            self.assert_table_configs_equal(jte_table, table)
            self.assert_table_configs_equal(keras_table, table)


if __name__ == "__main__":
    absltest.main()
