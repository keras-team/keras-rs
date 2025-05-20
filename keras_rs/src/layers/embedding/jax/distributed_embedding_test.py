import typing
from typing import Any, Tuple, Union

import jax
import jax.numpy as jnp
import keras
import numpy as np
import pytest
from absl.testing import absltest
from absl.testing import parameterized
from jax.experimental import layout as jax_layout
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn import table_stacking
from jax_tpu_embedding.sparsecore.utils import utils as jte_utils

from keras_rs.src.layers.embedding import test_utils as keras_test_utils
from keras_rs.src.layers.embedding.jax import config_conversion
from keras_rs.src.layers.embedding.jax import (
    distributed_embedding as jax_distributed_embedding,
)
from keras_rs.src.layers.embedding.jax import embedding_utils
from keras_rs.src.layers.embedding.jax import test_utils

keras.config.disable_traceback_filtering()


def _create_sparsecore_layout(
    sharding_axis: str = "sparsecore",
) -> keras.distribution.TensorLayout:
    # Distribute the embedding tables across all devices.
    all_devices = jax.devices()
    axes = [sharding_axis]
    device_mesh = keras.distribution.DeviceMesh(
        (len(all_devices),), axes, all_devices
    )
    sparsecore_layout = keras.distribution.TensorLayout(axes, device_mesh)
    # Custom sparsecore layout with tiling.
    sparsecore_layout._backend_layout = jax_layout.Layout(  # pylint: disable=protected-access
        jax_layout.DeviceLocalLayout(
            major_to_minor=(0, 1),
            _tiling=((8,),),
        ),
        jax.sharding.NamedSharding(
            device_mesh.backend_mesh, jax.sharding.PartitionSpec(axes)
        ),
    )
    return sparsecore_layout


def _num_sparsecores_per_device() -> int:
    if test_utils.has_sparsecores():
        return jte_utils.num_sparsecores_per_device()

    # Default to one for non-sparsecore tests.
    return 1


@pytest.mark.skipif(
    keras.backend.backend() != "jax",
    reason="Backend specific test",
)
class ShardedInitializerTest(parameterized.TestCase):
    @parameterized.product(
        initializer=[
            keras.initializers.RandomUniform(
                minval=-0.05, maxval=0.05, seed=10
            ),
            keras.initializers.Ones(),
            "zeros",
        ],
    )
    def test_wrap_and_call(
        self, initializer: Union[keras.initializers.Initializer, str]
    ):
        device_count = jax.device_count()
        layout = _create_sparsecore_layout()
        wrapped_initializer = jax_distributed_embedding.ShardedInitializer(
            initializer, layout
        )

        shape = (10 * device_count, 50)
        actual = wrapped_initializer(shape, dtype="float32")

        if isinstance(initializer, str):
            initializer = keras.initializers.get(initializer)
        expected = initializer(shape, dtype="float32")

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.sharding, layout.backend_layout.sharding)


@pytest.mark.skipif(
    keras.backend.backend() != "jax",
    reason="Backend specific test",
)
class StackedTableInitializerTest(parameterized.TestCase):
    def test_sharded_matches_unsharded(self):
        table_configs = (
            keras_test_utils.create_random_table_configs(
                count=1,
                initializer=keras.initializers.constant(10),
                name_prefix="table_a",
            )
            + keras_test_utils.create_random_table_configs(
                count=1,
                initializer=keras.initializers.constant(20),
                name_prefix="table_b",
            )
            + keras_test_utils.create_random_table_configs(
                count=1,
                initializer=keras.initializers.constant(30),
                name_prefix="table_c",
            )
        )
        feature_configs = keras_test_utils.create_random_feature_configs(
            table_configs=table_configs
        )

        device_count = jax.device_count()
        num_sc_per_device = _num_sparsecores_per_device()
        num_table_shards = device_count * num_sc_per_device

        # Convert to JAX and stack tables.
        feature_specs = config_conversion.keras_to_jte_feature_configs(
            feature_configs
        )
        feature_specs = typing.cast(
            list[embedding_spec.FeatureSpec], feature_specs
        )
        table_specs = {
            feature_spec.table_spec.name: feature_spec.table_spec
            for feature_spec in feature_specs
        }
        table_stacking.stack_tables(
            feature_specs,
            table_names=[table_config.name for table_config in table_configs],
            global_device_count=device_count,
            num_sc_per_device=num_sc_per_device,
        )
        stacked_table_spec = typing.cast(
            embedding_spec.StackedTableSpec,
            feature_specs[0].table_spec.stacked_table_spec,
        )

        layout = _create_sparsecore_layout()
        initializer = jax_distributed_embedding.StackedTableInitializer(
            table_specs, num_table_shards, layout
        )

        shape = (
            stacked_table_spec.stack_vocab_size,
            stacked_table_spec.stack_embedding_dim,
        )
        actual = initializer(shape, "float32")

        # Verify that tables are sharded correctly on devices.
        self.assertEqual(actual.sharding, layout.backend_layout.sharding)

        # Verify that tables match expected.
        expected_shape = (
            stacked_table_spec.stack_vocab_size,
            stacked_table_spec.stack_embedding_dim,
        )
        self.assertEqual(actual.shape, expected_shape)

        unsharded_tables = embedding_utils.unshard_and_unstack_tables(
            table_specs,
            {stacked_table_spec.stack_name: actual},
            num_table_shards,
        )
        individual_tables = test_utils.create_tables(table_specs)
        keras.tree.map_structure(
            np.testing.assert_array_equal,
            unsharded_tables,
            individual_tables,
        )

    def test_random_shards(self):
        table_configs = keras_test_utils.create_random_table_configs(
            count=3,
            initializer=keras.initializers.glorot_normal(),
        )
        feature_configs = keras_test_utils.create_random_feature_configs(
            table_configs=table_configs
        )
        feature_specs = config_conversion.keras_to_jte_feature_configs(
            feature_configs
        )
        feature_specs = typing.cast(
            list[embedding_spec.FeatureSpec], feature_specs
        )
        table_specs = {
            feature_spec.table_spec.name: feature_spec.table_spec
            for feature_spec in feature_specs
        }

        device_count = jax.device_count()
        num_sc_per_device = _num_sparsecores_per_device()
        num_table_shards = device_count * num_sc_per_device

        table_stacking.stack_tables(
            feature_specs,
            table_names=[
                table_spec.name for table_spec in table_specs.values()
            ],
            global_device_count=device_count,
            num_sc_per_device=num_sc_per_device,
        )
        stacked_table_spec = typing.cast(
            embedding_spec.StackedTableSpec,
            feature_specs[0].table_spec.stacked_table_spec,
        )

        layout = _create_sparsecore_layout()
        initializer = jax_distributed_embedding.StackedTableInitializer(
            table_specs, num_table_shards, layout
        )

        shape = (
            stacked_table_spec.stack_vocab_size,
            stacked_table_spec.stack_embedding_dim,
        )
        table = initializer(shape, "float32")

        # Check that all shards are different.
        sharded = table.reshape(
            num_table_shards, shape[0] // num_table_shards, shape[1]
        )
        for i in range(num_table_shards):
            for j in range(i + 1, num_table_shards):
                self.assertFalse(np.array_equal(sharded[i], sharded[j]))

        # Check that repeated calls produce the same result.
        table2 = initializer(shape, "float32")
        np.testing.assert_array_equal(table2, table)

    def test_compilability(self):
        table_configs = keras_test_utils.create_random_table_configs(
            count=3,
            initializer=keras.initializers.glorot_normal(),
        )
        feature_configs = keras_test_utils.create_random_feature_configs(
            table_configs=table_configs
        )
        feature_specs = config_conversion.keras_to_jte_feature_configs(
            feature_configs
        )
        feature_specs = typing.cast(
            list[embedding_spec.FeatureSpec], feature_specs
        )
        table_specs = {
            feature_spec.table_spec.name: feature_spec.table_spec
            for feature_spec in feature_specs
        }

        device_count = jax.device_count()
        num_sc_per_device = _num_sparsecores_per_device()
        num_table_shards = device_count * num_sc_per_device

        table_stacking.stack_tables(
            feature_specs,
            table_names=[
                table_spec.name for table_spec in table_specs.values()
            ],
            global_device_count=device_count,
            num_sc_per_device=num_sc_per_device,
        )
        stacked_table_spec = typing.cast(
            embedding_spec.StackedTableSpec,
            feature_specs[0].table_spec.stacked_table_spec,
        )

        shape = (
            stacked_table_spec.stack_vocab_size,
            stacked_table_spec.stack_embedding_dim,
        )

        def my_initializer(shape: Tuple[int, int], dtype: Any):
            layout = _create_sparsecore_layout()
            initializer = jax_distributed_embedding.StackedTableInitializer(
                table_specs, num_table_shards, layout
            )
            return initializer(shape, dtype)

        jit_initializer = jax.jit(
            my_initializer, static_argnames=["shape", "dtype"]
        )
        jit_table = jit_initializer(shape, "float32")

        table = my_initializer(shape, "float32")
        np.testing.assert_array_equal(jit_table, table)


@pytest.mark.skipif(
    keras.backend.backend() != "jax",
    reason="Backend specific test",
)
class DistributedEmbeddingLayerTest(parameterized.TestCase):
    @parameterized.product(
        ragged=[True, False],
        combiner=["sum", "mean", "sqrtn"],
        table_stacking=[
            "auto",
            ["table:0", "table:1"],
            [["table:0", "table:1", "table:2"]],
        ],
        jit=[True, False],
    )
    def test_call(
        self,
        ragged: bool,
        combiner: str,
        table_stacking: Union[str, list[str], list[list[str]]],
        jit: bool,
    ):
        if ragged and not test_utils.has_sparsecores():
            self.skipTest(
                "Ragged inputs are only supported on sparsecore devices."
            )

        table_configs = keras_test_utils.create_random_table_configs(
            combiner=combiner, seed=10
        )
        feature_configs = keras_test_utils.create_random_feature_configs(
            table_configs=table_configs, seed=20
        )
        layer = jax_distributed_embedding.DistributedEmbedding(
            feature_configs, table_stacking=table_stacking
        )

        # Trigger layer.build(...) to initialize tables.
        sample_ids, sample_weights = keras_test_utils.create_random_samples(
            feature_configs, ragged=ragged, seed=0
        )
        inputs = layer.preprocess(sample_ids, sample_weights)
        _ = layer(inputs)

        # Generate tables for test.
        seed = keras.random.SeedGenerator(40)
        tables = {
            table_config.name: keras.random.uniform(
                shape=(
                    table_config.vocabulary_size,
                    table_config.embedding_dim,
                ),
                minval=-5,
                maxval=5,
                dtype="float32",
                seed=seed,
            )
            for table_config in table_configs
        }
        layer.set_embedding_tables(tables)

        # Generate random inputs.
        sample_ids, sample_weights = keras_test_utils.create_random_samples(
            feature_configs, ragged=ragged, seed=30
        )
        inputs = layer.preprocess(sample_ids, sample_weights)
        call_fn = jax.jit(layer) if jit else layer
        outputs = call_fn(inputs)

        tables = layer.get_embedding_tables()
        expected_outputs = keras_test_utils.compute_expected_lookup(
            feature_configs, tables, sample_ids, sample_weights
        )

        keras.tree.map_structure(
            lambda a, b: np.testing.assert_allclose(a, b, atol=1e-5),
            outputs,
            expected_outputs,
        )

    @parameterized.product(
        ragged=[True, False],
        table_stacking=[
            "auto",
            [["table:0", "table:1", "table:2"]],
        ],
    )
    def test_fit(
        self,
        ragged: bool,
        table_stacking: Union[str, list[str], list[list[str]]],
    ):
        if ragged and not test_utils.has_sparsecores():
            self.skipTest(
                "Ragged inputs are only supported on sparsecore devices."
            )

        # Set global distribution to ensure optimizer variables are
        # replicated across all devices by default.
        keras.distribution.set_distribution(keras.distribution.DataParallel())

        table_configs = keras_test_utils.create_random_table_configs(
            max_vocabulary_size=64,
            max_embedding_dim=8,
            optimizer=keras.optimizers.SGD(learning_rate=0.1),
            seed=10,
        )
        feature_configs = keras_test_utils.create_random_feature_configs(
            table_configs=table_configs,
            batch_size=16,
            seed=20,
        )

        # Create tables for generating labels.
        seed = keras.random.SeedGenerator(40)
        tables = {
            table_config.name: keras.random.uniform(
                shape=(
                    table_config.vocabulary_size,
                    table_config.embedding_dim,
                ),
                minval=-5,
                maxval=5,
                dtype="float32",
                seed=seed,
            )
            for table_config in table_configs
        }

        # Fit and evaluate.
        def loss_fn(y_true, y_pred):
            return jnp.mean(jnp.square(y_true - y_pred))

        layer = jax_distributed_embedding.DistributedEmbedding(
            feature_configs, table_stacking=table_stacking
        )
        model = keras.Sequential([layer])
        model.compile(jit_compile=True, loss=loss_fn)

        # Dataset on which to evaluate the model.
        # We want to ensure the loss decreases before/after training.
        evaluation_dataset = keras_test_utils.RandomInputSampleDataset(
            feature_configs,
            tables,
            ragged=ragged,
            num_batches=10,
            seed=312,
            preprocessor=lambda inputs, weights: layer.preprocess(
                inputs, weights, training=True
            ),
        )

        # Warm-up the preprocessor to adjust max IDs / buffer sizes.
        for i in range(len(evaluation_dataset)):
            _ = evaluation_dataset[i]

        # Evaluate initial loss.
        loss_before = model.evaluate(evaluation_dataset)

        # Fit model to different dataset.
        training_dataset = keras_test_utils.RandomInputSampleDataset(
            feature_configs,
            tables,
            ragged=ragged,
            num_batches=100,
            seed=42,
            preprocessor=lambda inputs, weights: layer.preprocess(
                inputs, weights, training=True
            ),
        )
        model.fit(training_dataset, epochs=2)

        # Evaluate final loss.
        loss_after = model.evaluate(evaluation_dataset)
        np.testing.assert_array_less(loss_after, loss_before)


if __name__ == "__main__":
    absltest.main()
