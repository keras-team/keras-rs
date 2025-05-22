import functools
import typing
from typing import Any, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import keras
import numpy as np
import pytest
from absl.testing import absltest
from absl.testing import parameterized
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn import table_stacking
from jax_tpu_embedding.sparsecore.utils import utils as jte_utils

from keras_rs.src.layers.embedding.jax import embedding_lookup
from keras_rs.src.layers.embedding.jax import embedding_utils
from keras_rs.src.layers.embedding.jax import test_utils

shard_map = jax.experimental.shard_map.shard_map

Shape = Tuple[int, ...]
T = TypeVar("T")
Nested = Union[T, Sequence[T], Mapping[str, T]]


class TableInfo:
    name: str
    vocabulary_size: int
    embedding_size: int
    max_ids_per_partition: int
    max_unique_ids_per_partition: int

    def __init__(
        self,
        name: str,
        vocabulary_size: int,
        embedding_size: int,
        max_ids_per_partition: int = 128,
        max_unique_ids_per_partition: int = 128,
    ):
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.max_ids_per_partition = max_ids_per_partition
        self.max_unique_ids_per_partition = min(
            max_unique_ids_per_partition, max_ids_per_partition
        )


@pytest.mark.skipif(
    keras.backend.backend() != "jax",
    reason="Backend specific test",
)
class EmbeddingLookupTest(parameterized.TestCase):
    def assert_allclose(self, a: Any, b: Any, rtol=1e-7, atol=0) -> None:
        keras.tree.map_structure(
            functools.partial(
                np.testing.assert_allclose, rtol=rtol, atol=atol, strict=True
            ),
            a,
            b,
        )

    def _create_test_tables(
        self,
        table_info: Optional[Nested[TableInfo]],
        optimizer: Optional[embedding_spec.OptimizerSpec] = None,
        initializer: Optional[jax.nn.initializers.Initializer] = None,
    ) -> dict[str, embedding_spec.TableSpec]:
        return keras.tree.map_structure(
            lambda info: test_utils.create_table_spec(
                info.name,
                info.vocabulary_size,
                info.embedding_size,
                info.max_ids_per_partition,
                info.max_unique_ids_per_partition,
                initializer=initializer,
                optimizer=optimizer,
            ),
            table_info,
        )

    def _create_test_features(
        self,
        batch_size: int,
        sample_size: int,
        table_specs: Nested[embedding_spec.TableSpec],
    ) -> Nested[embedding_spec.FeatureSpec]:
        return keras.tree.map_structure(
            lambda table_spec: test_utils.create_feature_spec(
                table_spec.name + "_feature",
                table_spec,
                batch_size,
                sample_size,
            ),
            table_specs,
        )

    def _create_table_and_feature_specs(
        self,
        table_initializer: Optional[jax.nn.initializers.Initializer] = None,
        optimizer: Optional[embedding_spec.OptimizerSpec] = None,
    ):
        table_specs = self._create_test_tables(
            {
                "a": TableInfo("table_a", vocabulary_size=32, embedding_size=6),
                "b": TableInfo("table_b", vocabulary_size=65, embedding_size=7),
                "c": TableInfo("table_c", vocabulary_size=17, embedding_size=8),
            },
            initializer=table_initializer,
            optimizer=optimizer,
        )

        # Prepare sharded feature samples.
        feature_specs = self._create_test_features(
            batch_size=16,
            table_specs=table_specs,
            sample_size=10,
        )
        feature_specs = typing.cast(
            dict[str, embedding_spec.FeatureSpec], feature_specs
        )
        # Add a second feature to one of the tables.
        feature_specs["a2"] = test_utils.create_feature_spec(
            "table_a_feature_2",
            table_specs["a"],
            batch_size=16,
            sample_size=20,
        )
        return table_specs, feature_specs

    @parameterized.product(
        ragged=[True, False],
        stacked=[True, False],
    )
    def test_forward_pass(self, ragged: bool, stacked: bool):
        if not test_utils.has_sparsecores():
            self.skipTest("Test requires sparsecores.")

        devices = jax.devices()
        mesh = jax.sharding.Mesh(devices, "x")
        device_count = jax.device_count()
        local_device_count = jax.local_device_count()
        num_sc_per_device = jte_utils.num_sparsecores_per_device()
        num_table_shards = device_count * num_sc_per_device

        table_specs, feature_specs = self._create_table_and_feature_specs()
        if stacked:
            all_tables = [
                table_spec.name for table_spec in table_specs.values()
            ]
            table_stacking.stack_tables(
                feature_specs,
                table_names=all_tables,
                global_device_count=device_count,
                num_sc_per_device=num_sc_per_device,
            )
        else:
            embedding.prepare_feature_specs_for_training(
                feature_specs, device_count, num_sc_per_device
            )

        # Generate random tables and samples.
        tables = test_utils.create_tables(table_specs)
        feature_samples = test_utils.generate_feature_samples(
            feature_specs,
            ragged=ragged,
            max_samples=5,
        )

        # Shard tables/samples and send to device.
        config = embedding_lookup.EmbeddingLookupConfiguration(
            feature_specs, mesh=mesh, sharding_axis="x"
        )
        sharded_tables = test_utils.stack_shard_and_put_tables(
            table_specs,
            tables,
            num_table_shards,
            jax.sharding.NamedSharding(mesh, config.table_partition),
        )
        sharded_samples, _ = embedding_utils.stack_and_shard_samples(
            feature_specs,
            feature_samples,
            local_device_count,
            device_count,
            num_sc_per_device,
        )
        sharded_samples = jax.device_put(
            sharded_samples,
            jax.sharding.NamedSharding(mesh, config.samples_partition),
        )

        # Add pseudo gradients to the inputs.
        embedding_variables = jax.tree.map(
            lambda table: (table, None),
            sharded_tables,
        )

        # Run the forward pass.
        activations = embedding_lookup.embedding_lookup(
            config, sharded_samples, embedding_variables
        )
        expected = test_utils.compute_expected_lookup(
            feature_specs,
            feature_samples,
            table_specs,
            tables,
        )
        self.assert_allclose(activations, expected, atol=1e-6)

    @parameterized.product(
        ragged=[True, False],
        stacked=[True, False],
        num_model_shards=[1, 2, 4, 8],
    )
    def test_model_sharding(
        self, ragged: bool, stacked: bool, num_model_shards: int
    ):
        if not test_utils.has_sparsecores():
            self.skipTest("Test requires sparsecores.")

        if num_model_shards > jax.device_count():
            self.skipTest("Not enough devices for model shards")

        # The model does not seem to work unless we shard across all devices.
        num_data_shards = 1  # jax.device_count() // num_model_shards
        mesh = jax.make_mesh(
            (num_model_shards, num_data_shards), ["model", "data"]
        )
        num_sc_per_device = jte_utils.num_sparsecores_per_device()
        num_table_shards = num_model_shards * num_sc_per_device

        jax.config.update("jax_traceback_filtering", "off")

        table_specs, feature_specs = self._create_table_and_feature_specs()
        if stacked:
            all_tables = [
                table_spec.name for table_spec in table_specs.values()
            ]
            table_stacking.stack_tables(
                feature_specs,
                table_names=all_tables,
                global_device_count=num_model_shards,
                num_sc_per_device=num_sc_per_device,
            )
        else:
            embedding.prepare_feature_specs_for_training(
                feature_specs, num_model_shards, num_sc_per_device
            )

        # Generate random tables and samples.
        tables = test_utils.create_tables(table_specs)
        feature_samples = test_utils.generate_feature_samples(
            feature_specs,
            ragged=ragged,
            max_samples=5,
        )

        # Shard tables/samples and send to device.
        config = embedding_lookup.EmbeddingLookupConfiguration(
            feature_specs,
            mesh=mesh,
            table_partition=jax.sharding.PartitionSpec("model", None),
            samples_partition=jax.sharding.PartitionSpec("model"),
            # samples_partition=jax.sharding.PartitionSpec("data"),
        )
        sharded_tables = test_utils.stack_shard_and_put_tables(
            table_specs,
            tables,
            num_table_shards,
            jax.sharding.NamedSharding(mesh, config.table_partition),
        )
        sharded_samples, _ = embedding_utils.stack_and_shard_samples(
            feature_specs,
            feature_samples,
            num_model_shards,
            # num_data_shards,
            num_model_shards,
            num_sc_per_device,
        )

        # Send samples to device.
        sharded_samples = jax.device_put(
            sharded_samples,
            jax.sharding.NamedSharding(mesh, config.samples_partition),
        )

        # Add pseudo gradients to the inputs.
        embedding_variables = jax.tree.map(
            lambda table: (table, None),
            sharded_tables,
        )

        # Run the forward pass.
        activations = embedding_lookup.embedding_lookup(
            config, sharded_samples, embedding_variables
        )
        expected = test_utils.compute_expected_lookup(
            feature_specs,
            feature_samples,
            table_specs,
            tables,
        )
        self.assert_allclose(activations, expected, atol=1e-6)

    @parameterized.product(
        ragged=[True, False],
        stacked=[True, False],
        optimizer=[
            embedding_spec.SGDOptimizerSpec(0.001),
            embedding_spec.SGDOptimizerSpec(0.1),
            embedding_spec.AdagradOptimizerSpec(
                learning_rate=0.01, initial_accumulator_value=0.1
            ),
        ],
    )
    def test_backward_pass(
        self,
        ragged: bool,
        stacked: bool,
        optimizer: embedding_spec.OptimizerSpec,
    ):
        if not test_utils.has_sparsecores():
            self.skipTest("Test requires sparsecores.")

        devices = jax.devices()
        mesh = jax.sharding.Mesh(devices, "x")
        device_count = jax.device_count()
        local_device_count = jax.local_device_count()
        num_sc_per_device = jte_utils.num_sparsecores_per_device()
        num_table_shards = device_count * num_sc_per_device

        # Prepare tables.
        table_specs, feature_specs = self._create_table_and_feature_specs(
            optimizer=optimizer
        )
        if stacked:
            all_tables = [
                table_spec.name for table_spec in table_specs.values()
            ]
            table_stacking.stack_tables(
                feature_specs,
                table_names=all_tables,
                global_device_count=device_count,
                num_sc_per_device=num_sc_per_device,
            )
        else:
            embedding.prepare_feature_specs_for_training(
                feature_specs, device_count, num_sc_per_device
            )

        # Generate random tables and samples.
        table_and_slot_variables = test_utils.create_table_and_slot_variables(
            table_specs
        )
        feature_samples = test_utils.generate_feature_samples(
            feature_specs,
            ragged=ragged,
            max_samples=5,
        )

        # Shard tables and send to device.
        config = embedding_lookup.EmbeddingLookupConfiguration(
            feature_specs, mesh=mesh, sharding_axis="x"
        )
        sharded_table_and_slot_variables = (
            test_utils.stack_shard_and_put_tables(
                table_specs,
                table_and_slot_variables,
                num_table_shards,
                jax.sharding.NamedSharding(mesh, config.table_partition),
            )
        )

        # Shard samples for lookup query.
        sharded_samples, _ = embedding_utils.stack_and_shard_samples(
            feature_specs,
            feature_samples,
            local_device_count,
            device_count,
            num_sc_per_device,
        )

        # Create random gradients for the backward pass.
        activation_grads = jax.tree.map(
            lambda feature_spec: jax.random.uniform(
                key=jax.random.key(10), shape=feature_spec.output_shape
            ),
            feature_specs,
        )

        # Compute the updated tables and gradients, unstacked and unsharded.
        # NOTE: the following returns the updated tables rather than actual
        #       gradients.
        _, updated_stacked_tables, _ = embedding_lookup.embedding_lookup_bwd(
            config,
            res=(sharded_samples, sharded_table_and_slot_variables, None),
            gradients=activation_grads,
        )
        updated_tables_and_slots = embedding_utils.unshard_and_unstack_tables(
            table_specs, updated_stacked_tables, num_table_shards
        )

        expected = test_utils.compute_expected_updates(
            feature_specs,
            feature_samples,
            activation_grads,
            table_specs,
            table_and_slot_variables,
        )
        self.assert_allclose(updated_tables_and_slots, expected, atol=1e-6)

    @parameterized.product(
        ragged=[True, False],
        stacked=[True, False],
        optimizer=[
            embedding_spec.SGDOptimizerSpec(0.001),
            embedding_spec.SGDOptimizerSpec(0.1),
            embedding_spec.AdagradOptimizerSpec(
                learning_rate=0.01, initial_accumulator_value=0.1
            ),
        ],
    )
    def test_autograd(
        self,
        ragged: bool,
        stacked: bool,
        optimizer: embedding_spec.OptimizerSpec,
    ):
        if not test_utils.has_sparsecores():
            self.skipTest("Test requires sparsecores.")

        devices = jax.devices()
        mesh = jax.sharding.Mesh(devices, "x")
        device_count = jax.device_count()
        local_device_count = jax.local_device_count()
        num_sc_per_device = jte_utils.num_sparsecores_per_device()
        num_table_shards = device_count * num_sc_per_device

        table_specs, feature_specs = self._create_table_and_feature_specs(
            optimizer=optimizer
        )
        if stacked:
            all_tables = [
                table_spec.name for table_spec in table_specs.values()
            ]
            table_stacking.stack_tables(
                feature_specs,
                table_names=all_tables,
                global_device_count=device_count,
                num_sc_per_device=num_sc_per_device,
            )
        else:
            embedding.prepare_feature_specs_for_training(
                feature_specs, device_count, num_sc_per_device
            )

        # Generate random tables and samples.
        table_and_slot_variables = test_utils.create_table_and_slot_variables(
            table_specs
        )
        feature_samples = test_utils.generate_feature_samples(
            feature_specs,
            ragged=ragged,
            max_samples=5,
        )

        # Shard tables and send to device.
        config = embedding_lookup.EmbeddingLookupConfiguration(
            feature_specs, mesh=mesh, sharding_axis="x"
        )
        sharded_table_and_slot_variables = (
            test_utils.stack_shard_and_put_tables(
                table_specs,
                table_and_slot_variables,
                num_table_shards,
                jax.sharding.NamedSharding(mesh, config.table_partition),
            )
        )
        sharded_table_and_slot_variables = typing.cast(
            dict[str, Tuple[jax.Array, ...]], sharded_table_and_slot_variables
        )

        # Shard samples for lookup query.
        sharded_samples, _ = embedding_utils.stack_and_shard_samples(
            feature_specs,
            feature_samples,
            local_device_count,
            device_count,
            num_sc_per_device,
        )
        sharded_samples = jax.device_put(
            sharded_samples,
            jax.sharding.NamedSharding(mesh, config.samples_partition),
        )

        # Generate random dense matrices for use in a predict function.
        keys = keras.tree.unflatten_as(
            feature_specs,
            jnp.unstack(
                jax.random.split(
                    jax.random.key(0), len(keras.tree.flatten(feature_specs))
                )
            ),
        )
        dense_tables = keras.tree.map_structure(
            lambda feature_spec, key: jax.random.uniform(
                key=key,
                shape=(feature_spec.table_spec.embedding_dim, 1),
                dtype=jnp.float32,
            ),
            feature_specs,
            keys,
        )

        # Fake predict.
        def predict_fn(params, lookups):
            lookup_tables = params["lookup_tables"]
            dense_layer = params["dense_layer"]
            activations = embedding_lookup.embedding_lookup(
                config, lookups, lookup_tables
            )
            logits = sum(
                keras.tree.flatten(
                    jax.tree.map(jnp.matmul, activations, dense_layer)
                )
            )
            return logits

        # Fake loss.
        def loss_fn(params, lookups, labels):
            logits = predict_fn(params, lookups)
            loss = jnp.mean(jnp.square(logits - labels))
            return loss

        # Fake labels.
        model_params = {
            "lookup_tables": sharded_table_and_slot_variables,
            "dense_layer": dense_tables,
        }
        logits = predict_fn(model_params, sharded_samples)
        # Perturb logits slightly to get a non-zero gradient.
        labels = logits + 0.001 * jax.random.uniform(
            key=jax.random.key(0), shape=logits.shape
        )

        # Apply forward function and compute gradients.
        train_step_fn = jax.value_and_grad(loss_fn, allow_int=True)
        _, grads = train_step_fn(model_params, sharded_samples, labels)
        dense_grads = grads["dense_layer"]
        lookup_grads = grads["lookup_tables"]

        # Recover unstacked and unsharded gradients.
        updated_tables_and_slots = embedding_utils.unshard_and_unstack_tables(
            table_specs, lookup_grads, num_table_shards
        )

        # Compute embedding activation residual.
        activations = embedding_lookup.embedding_lookup(
            config, sharded_samples, sharded_table_and_slot_variables
        )
        lookup_res = keras.tree.map_structure(
            jnp.matmul,
            activations,
            dense_grads,
        )

        # Compute expected updated tables.
        expected_tables_and_slots = test_utils.compute_expected_updates(
            feature_specs,
            feature_samples,
            lookup_res,
            table_specs,
            table_and_slot_variables,
        )
        self.assert_allclose(
            updated_tables_and_slots, expected_tables_and_slots, atol=1e-6
        )


if __name__ == "__main__":
    absltest.main()
