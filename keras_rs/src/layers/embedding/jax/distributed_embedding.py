"""JAX implementation of the TPU embedding layer."""

import math
import typing
from typing import Any, Mapping, Sequence, Union

import jax
import keras
import numpy as np
from jax import numpy as jnp
from jax.experimental import layout as jax_layout
from jax.experimental import multihost_utils
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn import (
    table_stacking as jte_table_stacking,
)
from jax_tpu_embedding.sparsecore.utils import utils as jte_utils

from keras_rs.src import types
from keras_rs.src.layers.embedding import base_distributed_embedding
from keras_rs.src.layers.embedding import distributed_embedding_config as config
from keras_rs.src.layers.embedding.jax import config_conversion
from keras_rs.src.layers.embedding.jax import (
    embedding_lookup as jte_embedding_lookup,
)
from keras_rs.src.layers.embedding.jax import embedding_utils
from keras_rs.src.types import Nested
from keras_rs.src.utils import keras_utils

ArrayLike = Union[np.ndarray[Any, Any], jax.Array]
FeatureConfig = config.FeatureConfig
shard_map = jax.experimental.shard_map.shard_map  # type: ignore[attr-defined]


def _get_partition_spec(
    layout: (
        keras.distribution.TensorLayout
        | jax_layout.Format
        | jax.sharding.NamedSharding
        | jax.sharding.PartitionSpec
    ),
) -> Any:
    """Extracts the partition spec from a layout or sharding."""
    if isinstance(layout, keras.distribution.TensorLayout):
        layout = layout.backend_layout

    if isinstance(layout, jax_layout.Format):
        layout = layout.sharding

    if isinstance(layout, jax.sharding.NamedSharding):
        layout = layout.spec

    return layout


class ShardedInitializer(keras.initializers.Initializer):
    """Wraps an initializer to prepare for use with embedding tables.

    Jit-compiles the function and applies optimal output sharding to
    allow initialization on device.
    """

    def __init__(
        self,
        initializer: keras.initializers.Initializer | str,
        layout: keras.distribution.TensorLayout | None,
    ):
        if isinstance(initializer, str):
            initializer = keras.initializers.get(initializer)

        self._initializer = initializer
        self._layout = layout

    def __call__(
        self, shape: types.Shape, dtype: types.DType | None = None
    ) -> jax.Array:
        if self._layout is not None:
            compiled_initializer = jax.jit(
                self._initializer,
                out_shardings=self._layout.backend_layout,
                static_argnames=["shape", "dtype"],
            )
            output: jax.Array = compiled_initializer(shape, dtype)
            return output

        output = self._initializer(shape, dtype)
        return output


class StackedTableInitializer(keras.initializers.Initializer):
    """Initializes a single stacked table from multiple table initializers."""

    def __init__(
        self,
        table_specs: Nested[embedding_spec.TableSpec],
        num_shards: int,
        layout: keras.distribution.TensorLayout,
        seed: int | keras.random.SeedGenerator | jax.Array = 0,
    ):
        # Sort table specs so we can simply concatenate them when assembling the
        # stacked table.
        self._table_specs = sorted(
            keras.tree.flatten(table_specs),
            key=lambda table_spec: (
                table_spec.setting_in_stack.row_offset_in_shard,
            ),
        )
        self._num_shards = num_shards
        self._layout = layout
        self._key = keras.src.backend.jax.random.jax_draw_seed(seed)

    def _initialize_shard(
        self,
        keys: jax.Array,
        shape: tuple[int, int],
        dtype: Any,
        num_shards_per_device: int,
    ) -> jax.Array:
        """Initializes a single shard of a stacked table."""
        del shape  # Unused.
        table_shards: list[jax.Array] = []
        # NOTE: the following ignores padding, rotations in shard, and
        # mod-sharding, assuming all initializers are shard-independent.
        for i in range(num_shards_per_device):
            for j, table_spec in enumerate(self._table_specs):
                setting_in_stack = table_spec.setting_in_stack
                table_shard_shape = (
                    setting_in_stack.padded_vocab_size // self._num_shards,
                    setting_in_stack.padded_embedding_dim,
                )
                initializer = table_spec.initializer
                table_shards.append(
                    initializer(keys[i, j], table_shard_shape, dtype)
                )

        return jnp.concatenate(table_shards, axis=0)

    def __call__(
        self, shape: types.Shape, dtype: types.DType | None = None
    ) -> jax.Array:
        stacked_table_spec = typing.cast(
            embedding_spec.StackedTableSpec,
            self._table_specs[0].stacked_table_spec,
        )

        # Input shape is governed by the table specs.
        assert shape == (
            stacked_table_spec.stack_vocab_size,
            stacked_table_spec.stack_embedding_dim,
        )

        layout = self._layout
        backend_layout = layout.backend_layout
        backend_mesh = layout.device_mesh.backend_mesh
        num_devices_along_axis = backend_mesh.shape[layout.axes[0]]
        num_shards_per_device = self._num_shards // num_devices_along_axis
        shard_shape = (
            stacked_table_spec.stack_vocab_size // num_devices_along_axis,
            stacked_table_spec.stack_embedding_dim,
        )

        sharded_initializer = jax.jit(
            shard_map(
                lambda keys: self._initialize_shard(
                    keys, shard_shape, dtype, num_shards_per_device
                ),
                mesh=backend_mesh,
                in_specs=_get_partition_spec(backend_layout),
                out_specs=_get_partition_spec(backend_layout),
            ),
            out_shardings=backend_layout,
        )

        keys = jax.random.split(
            self._key, (self._num_shards, len(self._table_specs))
        )
        # Try extracting seeds from the existing table initializers.
        for i, table_spec in enumerate(self._table_specs):
            initializer = table_spec.initializer
            if isinstance(
                initializer, config_conversion.WrappedKerasInitializer
            ):
                initializer_key = initializer.key()
                if initializer_key is not None:
                    col = jax.random.split(initializer_key, self._num_shards)
                    keys = keys.at[:, i].set(col)

        output: jax.Array = sharded_initializer(keys)
        return output


class DistributedEmbedding(base_distributed_embedding.DistributedEmbedding):
    """JAX implementation of the TPU embedding layer."""

    def _create_sparsecore_distribution(
        self, sparsecore_axis_name: str = "sparsecore"
    ) -> tuple[
        keras.distribution.ModelParallel, keras.distribution.TensorLayout
    ]:
        """SparseCore requires a specific layout.

        The mesh must be 1D, must use all TPUs available, and must shard all
        tables across all devices.

        Args:
            sparsecore_axis_name: The name of the sparsecore axis.

        Returns:
            A Keras distribution to use for all sparsecore operations.
        """
        all_devices = jax.devices()
        axes = [sparsecore_axis_name]
        device_mesh = keras.distribution.DeviceMesh(
            (len(all_devices),), axes, all_devices
        )
        sparsecore_layout = keras.distribution.TensorLayout(axes, device_mesh)
        # Custom sparsecore layout with tiling.
        LayoutClass = (
            jax_layout.Layout
            if jax.__version_info__ >= (0, 6, 3)
            else jax_layout.DeviceLocalLayout  # type: ignore
        )
        layout = (
            LayoutClass(major_to_minor=(0, 1), tiling=((8,),))  # type: ignore
            if jax.__version_info__ >= (0, 7, 1)
            else LayoutClass(major_to_minor=(0, 1), _tiling=((8,),))  # type: ignore
        )
        # pylint: disable-next=protected-access
        sparsecore_layout._backend_layout = jax_layout.Format(
            layout,  # type: ignore
            jax.sharding.NamedSharding(
                device_mesh.backend_mesh,
                jax.sharding.PartitionSpec(
                    axes  # type: ignore[no-untyped-call]
                ),
            ),
        )
        layout_map = keras.distribution.LayoutMap(device_mesh=device_mesh)
        path = self.path
        if path is None:
            # Layer hasn't been properly built yet.  Use current layer name.
            path = self.name
        layout_map[path + "/var"] = sparsecore_layout
        sparsecore_distribution = keras.distribution.ModelParallel(
            layout_map=layout_map
        )
        return sparsecore_distribution, sparsecore_layout

    def _add_sparsecore_weight(
        self,
        name: str,
        shape: tuple[int, int],
        initializer: jax.nn.initializers.Initializer,
        dtype: Any,
        overwrite_with_gradient: bool,
    ) -> keras.Variable:
        var = self.add_weight(
            name=name, shape=shape, initializer=initializer, dtype=dtype
        )
        var.overwrite_with_gradient = overwrite_with_gradient
        return var

    def _add_table_variable(
        self,
        table_specs: Sequence[embedding_spec.TableSpec],
        num_shards: int,
        add_slot_variables: bool,
    ) -> embedding.EmbeddingVariables:
        stacked_table_spec = typing.cast(
            embedding_spec.StackedTableSpec, table_specs[0].stacked_table_spec
        )
        optimizer = stacked_table_spec.optimizer
        num_slot_variables = optimizer.slot_variables_count()
        table_shape = (
            stacked_table_spec.stack_vocab_size,
            stacked_table_spec.stack_embedding_dim,
        )

        # Make a stacked embedding table initializer.
        table_initializers = [
            config_conversion.jax_to_keras_initializer(table_spec.initializer)
            for table_spec in table_specs
        ]
        # If all initializers are the same, we can use a single sharded
        # initializer. Otherwise, we need to interleave individual stacked table
        # shards.
        sparsecore_layout = self._sparsecore_layout
        stacked_table_initializer = ShardedInitializer(
            table_initializers[0], sparsecore_layout
        )
        if not all(
            initializer == table_initializers[0]
            for initializer in table_initializers
        ):
            stacked_table_initializer = StackedTableInitializer(
                table_specs, num_shards, sparsecore_layout
            )

        variable_name = f"var:{stacked_table_spec.stack_name}:table"
        table_variable = self._add_sparsecore_weight(
            name=variable_name,
            shape=table_shape,
            initializer=stacked_table_initializer,
            dtype="float32",
            overwrite_with_gradient=True,
        )

        slot_variables = None
        if add_slot_variables:
            # All optimizers for a given stacked table are guaranteed to be the
            # same, so we can use a single sharded initializer for the entire
            # stacked table.
            slot_initializers = optimizer.slot_variables_initializers()
            # Try extracting field names from variables, otherwise just use the
            # count.
            slot_names = range(num_slot_variables)
            if hasattr(slot_initializers, "_fields"):
                slot_names = slot_initializers._fields

            slot_variables = tuple(
                self._add_sparsecore_weight(
                    name=f"{variable_name}:slot:{slot_name}",
                    shape=table_shape,
                    initializer=ShardedInitializer(
                        config_conversion.jax_to_keras_initializer(initializer),
                        sparsecore_layout,
                    ),
                    dtype=jnp.float32,
                    overwrite_with_gradient=True,
                )
                for slot_name, initializer in zip(slot_names, slot_initializers)
            )
            slot_variables = keras.tree.pack_sequence_as(
                slot_initializers, slot_variables
            )

        return embedding.EmbeddingVariables(table_variable, slot_variables)

    @keras_utils.no_automatic_dependency_tracking
    def _sparsecore_init(
        self,
        feature_configs: dict[str, FeatureConfig],
        table_stacking: str | Sequence[str] | Sequence[Sequence[str]],
    ) -> None:
        if not self._has_sparsecore():
            raise ValueError(
                "Not sparse cores available, cannot use explicit sparsecore"
                " placement."
            )

        self._sc_feature_configs = feature_configs
        self._sparsecore_built = False
        # Fill in any empty default settings.
        for feature_config in keras.tree.flatten(self._sc_feature_configs):
            if feature_config.table.initializer is None:
                table = feature_config.table
                table.initializer = keras.initializers.TruncatedNormal(
                    mean=0.0, stddev=1.0 / math.sqrt(float(table.embedding_dim))
                )

        # Actual stacking of tables is done in build() to ensure the
        # distribution is set up correctly.
        self._table_stacking = table_stacking

    def _sparsecore_build(
        self, input_shapes: Nested[types.Shape] | None = None
    ) -> None:
        self.sparsecore_build(input_shapes)

    @keras_utils.no_automatic_dependency_tracking
    def sparsecore_build(
        self, input_shapes: Nested[types.Shape] | None = None
    ) -> None:
        del input_shapes  # Unused.

        if self._sparsecore_built:
            return

        feature_specs = config_conversion.keras_to_jte_feature_configs(
            self._sc_feature_configs
        )

        # Distribution for sparsecore operations.
        sparsecore_distribution, sparsecore_layout = (
            self._create_sparsecore_distribution()
        )
        self._sparsecore_layout = sparsecore_layout
        self._sparsecore_distribution = sparsecore_distribution

        mesh = sparsecore_distribution.device_mesh.backend_mesh
        global_device_count = mesh.devices.size
        num_sc_per_device = jte_utils.num_sparsecores_per_device(
            mesh.devices.item(0)
        )
        # One table shard per global sparsecore.
        num_variable_shards = global_device_count * num_sc_per_device

        # Maybe stack tables.
        table_stacking = self._table_stacking
        if table_stacking is not None:
            if isinstance(table_stacking, str):
                if table_stacking == "auto":
                    jte_table_stacking.auto_stack_tables(
                        feature_specs, global_device_count, num_sc_per_device
                    )
                else:
                    raise ValueError(
                        f"Unsupported table stacking {table_stacking}, must be"
                        "None, 'auto', or sequences of table names to stack."
                    )
            else:
                if isinstance(table_stacking, list) and len(table_stacking) > 0:
                    elem = table_stacking[0]
                    # List of lists of table names.
                    if isinstance(elem, list):
                        for table_names in table_stacking:
                            jte_table_stacking.stack_tables(
                                feature_specs,
                                table_names,
                                global_device_count,
                                num_sc_per_device,
                            )
                    # Single list of table names.
                    elif isinstance(elem, str):
                        jte_table_stacking.stack_tables(
                            feature_specs,
                            table_stacking,
                            global_device_count,
                            num_sc_per_device,
                        )
                    else:
                        raise ValueError(
                            f"Unsupported table stacking {table_stacking}, "
                            "must be None, 'auto', or sequences of table names "
                            "to stack."
                        )

        # Adjust any non-stacked tables to prepare for training.
        embedding.prepare_feature_specs_for_training(
            feature_specs, global_device_count, num_sc_per_device
        )

        # Collect all stacked tables.
        table_specs = embedding.get_table_specs(feature_specs)
        table_stacks = jte_table_stacking.get_table_stacks(table_specs)

        # Update stacked table stats to max of values across involved tables.
        max_ids_per_partition = {}
        max_unique_ids_per_partition = {}
        required_buffer_size_per_device = {}
        id_drop_counters = {}
        for stack_name, stack in table_stacks.items():
            max_ids_per_partition[stack_name] = np.max(
                np.asarray(
                    [s.max_ids_per_partition for s in stack], dtype=np.int32
                )
            )
            max_unique_ids_per_partition[stack_name] = np.max(
                np.asarray(
                    [s.max_unique_ids_per_partition for s in stack],
                    dtype=np.int32,
                )
            )

            # Only set the suggested buffer size if set on any individual table.
            valid_buffer_sizes = [
                s.suggested_coo_buffer_size_per_device
                for s in stack
                if s.suggested_coo_buffer_size_per_device is not None
            ]
            if valid_buffer_sizes:
                required_buffer_size_per_device[stack_name] = np.max(
                    np.asarray(valid_buffer_sizes, dtype=np.int32)
                )

            id_drop_counters[stack_name] = 0

        aggregated_stats = embedding.SparseDenseMatmulInputStats(
            max_ids_per_partition=max_ids_per_partition,
            max_unique_ids_per_partition=max_unique_ids_per_partition,
            required_buffer_size_per_sc=required_buffer_size_per_device,
            id_drop_counters=id_drop_counters,
        )
        embedding.update_preprocessing_parameters(
            feature_specs,
            aggregated_stats,
            num_sc_per_device,
        )

        # Create variables for all stacked tables and slot variables.
        with sparsecore_distribution.scope():
            self._table_and_slot_variables = {
                table_name: self._add_table_variable(
                    table_stack,
                    add_slot_variables=self.trainable,
                    num_shards=num_variable_shards,
                )
                for table_name, table_stack in table_stacks.items()
            }

            # Create a step-counter variable for use in custom table gradients.
            # This must be a floating-point type so we can get a real gradient
            # for it. It will automatically be updated with each application of
            # the optimizer, since the next iteration is returned in the
            # gradient.
            sharded_zero_initializer = ShardedInitializer(
                "zeros",
                keras.distribution.TensorLayout(
                    [], sparsecore_layout.device_mesh
                ),
            )
            self._iterations = self.add_weight(
                shape=(),
                name="iteration",
                initializer=sharded_zero_initializer,
                dtype="float32",
                trainable=True,
            )
            self._iterations.overwrite_with_gradient = True

        self._config = jte_embedding_lookup.EmbeddingLookupConfiguration(
            feature_specs,
            mesh=mesh,
            table_partition=_get_partition_spec(sparsecore_layout),
            samples_partition=_get_partition_spec(sparsecore_layout),
            table_layout=sparsecore_layout.backend_layout,
        )

        self._sparsecore_built = True

    def _sparsecore_symbolic_preprocess(
        self,
        inputs: dict[str, types.Tensor],
        weights: dict[str, types.Tensor] | None,
        training: bool = False,
    ) -> dict[str, dict[str, embedding_utils.ShardedCooMatrix]]:
        """Allow preprocess(...) with `keras.Input`s.

        This is to support creating functional models via:
        ```python
        inputs = keras.Input(shape=(None), dtype="int32")
        weights = keras.Input(shape=(None), dtype="float32")
        preprocessed_inputs = distributed_embedding.preprocess(inputs, weights)
        outputs = distributed_embedding(preprocessed_inputs)
        model = keras.Model(inputs=preprocessed_inputs, outputs=outputs)
        ```

        Args:
            inputs: SparseCore path->tensor input ID's tensors.
            weights: Optional Sparsecore path->tensor input weights tensors.
            training: Whether the layer is training or not.

        Returns:
            Symbolic preprocessed input tensors to the layer/model.
        """
        # Arguments are currently ignored since the input shape is governed
        # by the stacked table configuration.
        del inputs, weights, training

        # Each stacked-table gets a ShardedCooMatrix.
        table_specs = embedding.get_table_specs(self._config.feature_specs)
        table_stacks = jte_table_stacking.get_table_stacks(table_specs)
        stacked_table_specs = {
            stack_name: stack[0].stacked_table_spec
            for stack_name, stack in table_stacks.items()
        }

        def _compute_table_output_spec(
            stacked_table_spec: embedding_spec.StackedTableSpec,
        ) -> embedding_utils.ShardedCooMatrix:
            # The true shape of the components in the ShardedCooMatrix depends
            # on the hardware configuration (# devices, sparsecores),
            # properties of the input data (# max IDs, unique IDs), and other
            # hints like a suggested internal buffer size.  Some of the
            # calculations are currently a bit in flux as we experiment with
            # memory trade-offs.  For the purposes of input/output sizes,
            # however, the size could be viewed as dynamic 1D without affecting
            # the output spec sizes.
            del stacked_table_spec
            return embedding_utils.ShardedCooMatrix(
                # Mark these as `Input`s since that's how they will be used when
                # constructing a functional Keras model.
                shard_starts=keras.Input(shape=tuple(), dtype="int32"),
                shard_ends=keras.Input(shape=tuple(), dtype="int32"),
                col_ids=keras.Input(shape=tuple(), dtype="int32"),
                row_ids=keras.Input(shape=tuple(), dtype="int32"),
                values=keras.Input(shape=tuple(), dtype="float32"),
            )

        preprocessed = keras.tree.map_structure(
            _compute_table_output_spec, stacked_table_specs
        )

        return {"inputs": preprocessed}

    def _sparsecore_preprocess(
        self,
        inputs: dict[str, types.Tensor],
        weights: dict[str, types.Tensor] | None,
        training: bool = False,
    ) -> dict[str, dict[str, embedding_utils.ShardedCooMatrix]]:
        if any(
            isinstance(x, jax.core.Tracer) for x in keras.tree.flatten(inputs)
        ):
            raise ValueError(
                "DistributedEmbedding.preprocess(...) does not support"
                " jit-compilation"
            )

        if not self._sparsecore_built:
            self._sparsecore_build()

        # Support symbolic KerasTensors (i.e. keras.Input).
        if any(
            isinstance(x, keras.KerasTensor) for x in keras.tree.flatten(inputs)
        ):
            return self._sparsecore_symbolic_preprocess(
                inputs, weights, training
            )

        samples = embedding_utils.create_feature_samples(
            self._config.feature_specs, inputs, weights
        )

        layout = self._sparsecore_layout
        mesh = layout.device_mesh.backend_mesh
        global_device_count = mesh.devices.size
        local_device_count = mesh.local_mesh.devices.size
        num_sc_per_device = jte_utils.num_sparsecores_per_device(
            mesh.devices.item(0)
        )

        preprocessed, stats = embedding_utils.stack_and_shard_samples(
            self._config.feature_specs,
            samples,
            local_device_count,
            global_device_count,
            num_sc_per_device,
        )

        if training:
            # Synchronize input statistics across all devices and update the
            # underlying stacked tables specs in the feature specs.

            # Aggregate stats across all processes/devices via pmax.
            all_stats = multihost_utils.process_allgather(stats)
            aggregated_stats = jax.tree.map(
                lambda x: jnp.max(x, axis=0), all_stats
            )

            # Check if stats changed enough to warrant action.
            stacked_table_specs = embedding.get_stacked_table_specs(
                self._config.feature_specs
            )
            changed = any(
                np.max(aggregated_stats.max_ids_per_partition[stack_name])
                > spec.max_ids_per_partition
                or np.max(
                    aggregated_stats.max_unique_ids_per_partition[stack_name]
                )
                > spec.max_unique_ids_per_partition
                or (
                    np.max(
                        aggregated_stats.required_buffer_size_per_sc[stack_name]
                    )
                    * num_sc_per_device
                )
                > (spec.suggested_coo_buffer_size_per_device or 0)
                for stack_name, spec in stacked_table_specs.items()
            )

            # Update configuration and repeat preprocessing if stats changed.
            if changed:
                embedding.update_preprocessing_parameters(
                    self._config.feature_specs,
                    aggregated_stats,
                    num_sc_per_device,
                )

                # Re-execute preprocessing with consistent input statistics.
                preprocessed, _ = embedding_utils.stack_and_shard_samples(
                    self._config.feature_specs,
                    samples,
                    local_device_count,
                    global_device_count,
                    num_sc_per_device,
                )

        return {"inputs": preprocessed}

    def _sparsecore_call(
        self,
        inputs: dict[str, types.Tensor],
        weights: dict[str, types.Tensor] | None = None,
        training: bool = False,
        **kwargs: Any,
    ) -> dict[str, types.Tensor]:
        assert weights is None

        if not self._sparsecore_built:
            self._sparsecore_build()

        table_and_slots = keras.tree.map_structure(
            lambda var: var.value, self._table_and_slot_variables
        )
        with self._sparsecore_distribution.scope():
            lookup_func = jax.jit(
                jte_embedding_lookup.embedding_lookup, static_argnames="config"
            )
            out: dict[str, types.Tensor] = lookup_func(
                self._config, inputs, table_and_slots, self._iterations.value
            )
            return out

    def set_embedding_tables(self, tables: Mapping[str, ArrayLike]) -> None:
        """Sets the embedding tables to specific (unsharded) values.

        Args:
          tables: Mapping of table name -> table values.
        """
        if "default_device" in self._placement_to_path_to_feature_config:
            self._default_device_set_tables(tables)

        if "sparsecore" in self._placement_to_path_to_feature_config:
            self._sparsecore_set_tables(tables)

    def _default_device_set_tables(
        self, tables: Mapping[str, ArrayLike]
    ) -> None:
        if not self.built:
            raise ValueError("Layer must first be built before setting tables.")

        if "default_device" in self._placement_to_path_to_feature_config:
            table_name_to_embedding_layer = {}
            for (
                path,
                feature_config,
            ) in self._placement_to_path_to_feature_config[
                "default_device"
            ].items():
                table_name_to_embedding_layer[feature_config.table.name] = (
                    self._default_device_embedding_layers[path]
                )

            for (
                table_name,
                embedding_layer,
            ) in table_name_to_embedding_layer.items():
                table_values = tables.get(table_name, None)
                if table_values is not None:
                    if embedding_layer.lora_enabled:
                        raise ValueError("Cannot set table if LoRA is enabled.")
                    # pylint: disable-next=protected-access
                    embedding_layer._embeddings.assign(table_values)

    def _sparsecore_set_tables(self, tables: Mapping[str, ArrayLike]) -> None:
        if not self._sparsecore_built:
            self._sparsecore_build()

        config = self._config
        num_table_shards = config.mesh.devices.size * config.num_sc_per_device
        table_specs = embedding.get_table_specs(config.feature_specs)
        sharded_tables = jte_table_stacking.stack_and_shard_tables(
            table_specs,
            tables,
            num_table_shards,
        )

        device_tables = jax.device_put(
            jax.tree.map(
                # Flatten shard dimension to allow auto-sharding to split the
                # array.
                lambda table: table.reshape((-1, table.shape[-1])),
                sharded_tables,
            ),
            self._sparsecore_layout.backend_layout,
        )

        # Assign stacked table variables to the device values.
        keras.tree.map_structure_up_to(
            device_tables,
            lambda embedding_variables,
            table_value: embedding_variables.table.assign(table_value),
            self._table_and_slot_variables,
            device_tables,
        )

    def _sparsecore_get_embedding_tables(self) -> dict[str, ArrayLike]:
        if not self._sparsecore_built:
            self.sparsecore_build()

        config = self._config
        num_table_shards = config.mesh.devices.size * config.num_sc_per_device
        table_specs = embedding.get_table_specs(config.feature_specs)

        # Extract only the table variables, not the gradient slot variables.
        table_variables = {
            name: jax.device_get(embedding_variables.table.value)
            for name, embedding_variables in (
                self._table_and_slot_variables.items()
            )
        }

        return typing.cast(
            dict[str, ArrayLike],
            jte_table_stacking.unshard_and_unstack_tables(
                table_specs, table_variables, num_table_shards
            ),
        )
