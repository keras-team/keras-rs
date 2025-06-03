"""Utility functions for manipulating JAX embedding tables and inputs."""

import collections
import dataclasses
import typing
from typing import Any, Mapping, NamedTuple, Sequence, TypeAlias, TypeVar

import jax
import numpy as np
from jax import numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn.embedding_spec import FeatureSpec
from jax_tpu_embedding.sparsecore.lib.nn.embedding_spec import StackedTableSpec
from jax_tpu_embedding.sparsecore.lib.nn.embedding_spec import TableSpec

from keras_rs.src.types import Nested

T = TypeVar("T")

# Any to support tf.Ragged without needing an explicit TF dependency.
ArrayLike: TypeAlias = jax.Array | np.ndarray | Any  # type: ignore
Shape: TypeAlias = tuple[int, ...]


class FeatureSamples(NamedTuple):
    tokens: ArrayLike
    weights: ArrayLike


class ShardedCooMatrix(NamedTuple):
    shard_starts: ArrayLike
    shard_ends: ArrayLike
    col_ids: ArrayLike
    row_ids: ArrayLike
    values: ArrayLike


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _default_stacked_table_spec(
    table_spec: TableSpec, num_shards: int, batch_size: int
) -> StackedTableSpec:
    return StackedTableSpec(
        stack_name=table_spec.name,
        stack_vocab_size=_round_up_to_multiple(
            table_spec.vocabulary_size, 8 * num_shards
        ),
        stack_embedding_dim=_round_up_to_multiple(table_spec.embedding_dim, 8),
        optimizer=table_spec.optimizer,
        combiner=table_spec.combiner,
        total_sample_count=batch_size,
        max_ids_per_partition=table_spec.max_ids_per_partition,
        max_unique_ids_per_partition=table_spec.max_unique_ids_per_partition,
    )


def _get_stacked_table_spec(
    table_spec: TableSpec, num_shards: int, batch_size: int = 0
) -> StackedTableSpec:
    return table_spec.stacked_table_spec or _default_stacked_table_spec(
        table_spec, num_shards, batch_size
    )


def pad_table(
    table_spec: TableSpec,
    table_values: jax.Array,
    num_shards: int,
    pad_value: jnp.float32 = jnp.nan,
) -> jax.Array:
    """Adds appropriate padding to a table to prepare for stacking.

    Args:
        table_spec: Table specification describing the table to pad.
        table_values: Table values array to pad.
        num_shards: Number of shards in the table (typically
            `global_device_count * num_sc_per_device`).
        pad_value: Value to use for padding.

    Returns:
        Padded table values.
    """
    vocabulary_size = table_spec.vocabulary_size
    embedding_dim = table_spec.embedding_dim
    padded_vocabulary_size = _round_up_to_multiple(
        vocabulary_size, 8 * num_shards
    )
    stack_embedding_dim = _get_stacked_table_spec(
        table_spec, num_shards
    ).stack_embedding_dim
    return jnp.pad(
        table_values,
        (
            (0, padded_vocabulary_size - vocabulary_size),
            (0, stack_embedding_dim - embedding_dim),
        ),
        constant_values=pad_value,
    )


def _stack_and_shard_table(
    stacked_table: jax.Array,
    table_spec: TableSpec,
    table: jax.Array,
    num_shards: int,
    pad_value: jnp.float32,
) -> jax.Array:
    """Stacks and shards a single table for use in sparsecore lookups."""
    padded_values = pad_table(table_spec, table, num_shards, pad_value)
    sharded_padded_vocabulary_size = padded_values.shape[0] // num_shards
    stack_embedding_dim = stacked_table.shape[-1]

    # Mod-shard vocabulary across devices.
    sharded_values = jnp.swapaxes(
        padded_values.reshape(-1, num_shards, stack_embedding_dim),
        0,
        1,
    )

    # Rotate shards.
    setting_in_stack = table_spec.setting_in_stack
    rotated_values = jnp.roll(
        sharded_values, setting_in_stack.shard_rotation, axis=0
    )

    # Insert table into the stack.
    table_row = setting_in_stack.row_offset_in_shard
    stacked_table = stacked_table.at[
        :, table_row : (table_row + sharded_padded_vocabulary_size), :
    ].set(rotated_values)

    return stacked_table


def stack_and_shard_tables(
    table_specs: Nested[TableSpec],
    tables: Nested[ArrayLike],
    num_shards: int,
    pad_value: jnp.float32 = jnp.nan,
) -> dict[str, Nested[jax.Array]]:
    """Stacks and shards tables for use in sparsecore lookups.

    Args:
        table_specs: Nested collection of unstacked table specifications.
        tables: Table values corresponding to the table_specs.
        num_shards: Number of shards in the table (typically
            `global_device_count * num_sc_per_device`).
        pad_value: Value to use for padding.

    Returns:
        A mapping of stacked table names to stacked table values.
    """

    # Gather stacked table information.
    stacked_table_map: dict[
        str,
        tuple[StackedTableSpec, list[TableSpec]],
    ] = {}

    def collect_stacked_tables(table_spec: TableSpec) -> None:
        stacked_table_spec = _get_stacked_table_spec(table_spec, num_shards)
        stacked_table_name = stacked_table_spec.stack_name
        if stacked_table_name not in stacked_table_map:
            stacked_table_map[stacked_table_name] = (stacked_table_spec, [])
        stacked_table_map[stacked_table_name][1].append(table_spec)

    _ = jax.tree.map(collect_stacked_tables, table_specs)

    table_map: dict[str, Nested[jax.Array]] = {}

    def collect_tables(table_spec: TableSpec, table: Nested[jax.Array]) -> None:
        table_map[table_spec.name] = table

    _ = jax.tree.map(collect_tables, table_specs, tables)

    stacked_tables: dict[str, Nested[jax.Array]] = {}
    for (
        stacked_table_spec,
        table_specs,
    ) in stacked_table_map.values():
        stack_vocab_size = stacked_table_spec.stack_vocab_size
        sharded_vocab_size = stack_vocab_size // num_shards
        stack_embedding_dim = stacked_table_spec.stack_embedding_dim

        # Allocate initial buffer.  The stacked table will be divided among
        # shards by splitting the vocabulary dimension:
        #   [ v, e ] -> [s, v/s, e]
        stacked_table_tree = jax.tree.map(
            lambda _: jnp.zeros(
                # pylint: disable-next=cell-var-from-loop, used only in loop body.
                shape=(num_shards, sharded_vocab_size, stack_embedding_dim),
                dtype=jnp.float32,
            ),
            table_map[table_specs[0].name],
        )

        for table_spec in table_specs:
            table_tree = table_map[table_spec.name]
            stacked_table_tree = jax.tree.map(
                lambda stacked_table, table: _stack_and_shard_table(
                    # pylint: disable-next=cell-var-from-loop, used only in loop body.
                    stacked_table,
                    # pylint: disable-next=cell-var-from-loop, used only in loop body.
                    table_spec,
                    table,
                    num_shards,
                    pad_value,
                ),
                stacked_table_tree,
                table_tree,
            )

        stacked_tables[stacked_table_spec.stack_name] = stacked_table_tree

    return stacked_tables


def _unshard_and_unstack_table(
    table_spec: TableSpec,
    stacked_table_tree: Nested[jax.Array],
    num_shards: int,
) -> Nested[jax.Array]:
    """Unshards and unstacks a single table."""
    vocabulary_size = table_spec.vocabulary_size
    embedding_dim = table_spec.embedding_dim

    def _unshard_and_unstack_single_table(
        table_spec: TableSpec, stacked_table: jax.Array
    ) -> jax.Array:
        stack_embedding_dim = stacked_table.shape[-1]

        # Maybe re-shape in case it was flattened.
        stacked_table = stacked_table.reshape(
            num_shards, -1, stack_embedding_dim
        )
        sharded_vocabulary_size = (
            _round_up_to_multiple(vocabulary_size, 8 * num_shards) // num_shards
        )

        # Extract padded values from the stacked table.
        setting_in_stack = table_spec.setting_in_stack
        row = setting_in_stack.row_offset_in_shard
        padded_values = stacked_table[
            :, row : (row + sharded_vocabulary_size), :
        ]

        # Un-rotate shards.
        padded_values = jnp.roll(
            padded_values, -setting_in_stack.shard_rotation, axis=0
        )

        # Un-mod-shard.
        padded_values = jnp.swapaxes(padded_values, 0, 1).reshape(
            -1, stack_embedding_dim
        )

        # Un-pad.
        return padded_values[:vocabulary_size, :embedding_dim]

    output: Nested[jax.Array] = jax.tree.map(
        lambda stacked_table: _unshard_and_unstack_single_table(
            table_spec, stacked_table
        ),
        stacked_table_tree,
    )
    return output


def unshard_and_unstack_tables(
    table_specs: Nested[TableSpec],
    stacked_tables: Mapping[str, Nested[jax.Array]],
    num_shards: int,
) -> Nested[jax.Array]:
    """Unshards and unstacks a collection of tables.

    Args:
        table_specs: Nested collection of unstacked table specifications.
        stacked_tables: Mapping of stacked table names to stacked table values.
        num_shards: Number of shards in the table (typically
            `global_device_count * num_sc_per_device`).

    Returns:
        A mapping of table names to unstacked table values.
    """
    output: Nested[jax.Array] = jax.tree.map(
        lambda table_spec: _unshard_and_unstack_table(
            table_spec,
            stacked_tables[
                _get_stacked_table_spec(table_spec, num_shards=1).stack_name
            ],
            num_shards,
        ),
        table_specs,
    )
    return output


def get_table_specs(feature_specs: Nested[FeatureSpec]) -> dict[str, TableSpec]:
    table_spec_map: dict[str, TableSpec] = {}
    flat_feature_specs, _ = jax.tree.flatten(feature_specs)
    for feature_spec in flat_feature_specs:
        table_spec = feature_spec.table_spec
        table_spec_map[table_spec.name] = table_spec
    return table_spec_map


def get_table_stacks(
    table_specs: Nested[TableSpec],
) -> dict[str, list[TableSpec]]:
    """Extracts lists of tables that are stacked together.

    Args:
        table_specs: Nested collection of table specifications.

    Returns:
        A mapping of stacked table names to lists of table specifications for
        each stack.
    """
    stacked_table_specs: dict[str, list[TableSpec]] = collections.defaultdict(
        list
    )
    flat_table_specs, _ = jax.tree.flatten(table_specs)
    for table_spec in flat_table_specs:
        table_spec = typing.cast(TableSpec, table_spec)
        stacked_table_spec = table_spec.stacked_table_spec
        if stacked_table_spec is not None:
            stacked_table_specs[stacked_table_spec.stack_name].append(
                table_spec
            )
        else:
            stacked_table_specs[table_spec.name].append(table_spec)

    return stacked_table_specs


def update_stacked_table_specs(
    feature_specs: Nested[FeatureSpec],
    max_ids_per_partition: Mapping[str, int],
    max_unique_ids_per_partition: Mapping[str, int],
) -> None:
    """Updates properties in the supplied feature specs.

    Args:
        feature_specs: Feature specs to update in-place.
        max_ids_per_partition: Mapping of table stack name to
            new `max_ids_per_partition` for the stack.
        max_unique_ids_per_partition: Mapping of table stack name to
            new `max_unique_ids_per_partition` for the stack.
    """
    # Collect table specs and stacked table specs.
    table_specs: dict[str, TableSpec] = {}
    for feature_spec in jax.tree.flatten(feature_specs)[0]:
        feature_spec = typing.cast(FeatureSpec, feature_spec)
        table_specs[feature_spec.table_spec.name] = feature_spec.table_spec

    stacked_table_specs: dict[str, StackedTableSpec] = {}
    for table_spec in table_specs.values():
        stacked_table_spec = typing.cast(
            StackedTableSpec, table_spec.stacked_table_spec
        )
        stacked_table_specs[stacked_table_spec.stack_name] = stacked_table_spec

    # Replace fields in the stacked_table_specs.
    stacked_table_specs = {
        stack_name: dataclasses.replace(
            stacked_table_spec,
            max_ids_per_partition=max_ids_per_partition[
                stacked_table_spec.stack_name
            ],
            max_unique_ids_per_partition=max_unique_ids_per_partition[
                stacked_table_spec.stack_name
            ],
        )
        for stack_name, stacked_table_spec in stacked_table_specs.items()
    }

    # Insert new stacked tables into tables.
    for table_spec in table_specs.values():
        stacked_table_spec = typing.cast(
            StackedTableSpec, table_spec.stacked_table_spec
        )
        table_spec.stacked_table_spec = stacked_table_specs[
            stacked_table_spec.stack_name
        ]


def convert_to_numpy(
    ragged_or_dense: np.ndarray[Any, Any] | Sequence[Sequence[Any]] | Any,
    dtype: Any,
) -> np.ndarray[Any, Any]:
    """Converts a ragged or dense list of inputs to a ragged/dense numpy array.

    The output is adjusted to be 2D.

    Args:
        ragged_or_dense: Input that is either already a numpy array, or nested
            sequence.
        dtype: Numpy dtype of output array.

    Returns:
        Corresponding numpy array.
    """
    if hasattr(ragged_or_dense, "numpy"):
        # Support tf.RaggedTensor and other TF input dtypes.
        if callable(getattr(ragged_or_dense, "numpy")):
            ragged_or_dense = ragged_or_dense.numpy()

    if isinstance(ragged_or_dense, jax.Array):
        ragged_or_dense = np.asarray(ragged_or_dense)

    if isinstance(ragged_or_dense, np.ndarray):
        # Convert 1D to 2D.
        if ragged_or_dense.dtype != np.ndarray and ragged_or_dense.ndim == 1:
            return ragged_or_dense.reshape(-1, 1).astype(dtype)

        # If dense, return converted dense type.
        if ragged_or_dense.dtype != np.ndarray:
            return ragged_or_dense.astype(dtype)

        # Ragged numpy array.
        return ragged_or_dense

    # Handle 1D sequence input.
    if not isinstance(ragged_or_dense[0], collections.abc.Sequence):
        return np.asarray(ragged_or_dense, dtype=dtype).reshape(-1, 1)

    # Assemble elements into an nd-array.
    counts = [len(vals) for vals in ragged_or_dense]
    if all([count == counts[0] for count in counts]):
        # Dense input.
        return np.asarray(ragged_or_dense, dtype=dtype)
    else:
        # Ragged input, convert to ragged numpy arrays.
        return np.array(
            [np.array(row, dtype=dtype) for row in ragged_or_dense],
            dtype=np.ndarray,
        )


def ones_like(
    ragged_or_dense: np.ndarray[Any, Any], dtype: Any = None
) -> np.ndarray[Any, Any]:
    """Creates an array of ones the same as as the input.

    This differs from traditional numpy in that a ragged input will lead to
    a resulting ragged array of ones, whereas np.ones_like(...) will instead
    only consider the outer array and return a 1D dense array of ones.

    Args:
        ragged_or_dense: The ragged or dense input whose shape and data-type
              define these same attributes of the returned array.
        dtype: The data-type of the returned array.

    Returns:
        An array of ones with the same shape as the input, and specified data
        type.
    """
    dtype = dtype or ragged_or_dense.dtype
    if ragged_or_dense.dtype == np.ndarray:
        # Ragged.
        return np.array(
            [np.ones_like(row, dtype=dtype) for row in ragged_or_dense],
            dtype=np.ndarray,
        )
    else:
        # Dense.
        return np.ones_like(ragged_or_dense, dtype=dtype)


def create_feature_samples(
    feature_structure: Nested[T],
    feature_ids: Nested[ArrayLike | Sequence[int] | Sequence[Sequence[int]]],
    feature_weights: None
    | (Nested[ArrayLike | Sequence[float] | Sequence[Sequence[float]]]),
) -> Nested[FeatureSamples]:
    """Constructs a collection of sample tuples from provided IDs and weights.

    Args:
        feature_structure: The nested structure of the inputs (typically
            `FeatureSpec`s).
        feature_ids: The feature IDs to use for the samples.
        feature_weights: The feature weights to use for the samples.  Defaults
            to ones if not provided.

    Returns:
        A nested collection of `FeatureSamples` corresponding to the input IDs
        and weights, for use in embedding lookups.
    """
    # Create numpy arrays from inputs.
    feature_ids = jax.tree.map(
        lambda _, ids: convert_to_numpy(ids, np.int32),
        feature_structure,
        feature_ids,
    )

    if feature_weights is None:
        # Make ragged or dense ones_like.
        feature_weights = jax.tree.map(
            lambda _, ids: ones_like(ids, np.float32),
            feature_structure,
            feature_ids,
        )
    else:
        feature_weights = jax.tree.map(
            lambda _, wgts: convert_to_numpy(wgts, np.float32),
            feature_structure,
            feature_weights,
        )

    # Assemble.
    def _create_feature_samples(
        sample_ids: np.ndarray[Any, Any],
        sample_weights: np.ndarray[Any, Any],
    ) -> FeatureSamples:
        return FeatureSamples(sample_ids, sample_weights)

    output: Nested[FeatureSamples] = jax.tree.map(
        lambda _, sample_ids, sample_weights: _create_feature_samples(
            sample_ids, sample_weights
        ),
        feature_structure,
        feature_ids,
        feature_weights,
    )
    return output


def stack_and_shard_samples(
    feature_specs: Nested[FeatureSpec],
    feature_samples: Nested[FeatureSamples],
    local_device_count: int,
    global_device_count: int,
    num_sc_per_device: int,
    static_buffer_size: int | Mapping[str, int] | None = None,
) -> tuple[dict[str, ShardedCooMatrix], embedding.SparseDenseMatmulInputStats]:
    """Prepares input samples for use in embedding lookups.

    Args:
        feature_specs: Nested collection of feature specifications.
        feature_samples: Nested collection of feature samples.
        local_device_count: Number of local JAX devices.
        global_device_count: Number of global JAX devices.
        num_sc_per_device: Number of sparsecores per device.
        static_buffer_size: The static buffer size to use for the samples.
            Defaults to None, in which case an upper-bound for the buffer size
            will be automatically determined.

    Returns:
        The preprocessed inputs, and statistics useful for updating FeatureSpecs
        based on the provided input data.
    """
    del static_buffer_size  # Currently ignored.
    flat_feature_specs, _ = jax.tree.flatten(feature_specs)

    feature_tokens = []
    feature_weights = []

    def collect_tokens_and_weights(
        feature_spec: FeatureSpec, samples: FeatureSamples
    ) -> None:
        del feature_spec
        feature_tokens.append(samples.tokens)
        feature_weights.append(samples.weights)

    jax.tree.map(collect_tokens_and_weights, feature_specs, feature_samples)

    preprocessed_inputs, stats = embedding.preprocess_sparse_dense_matmul_input(
        feature_tokens,
        feature_weights,
        flat_feature_specs,
        local_device_count=local_device_count,
        global_device_count=global_device_count,
        num_sc_per_device=num_sc_per_device,
        sharding_strategy="MOD",
        has_leading_dimension=False,
        allow_id_dropping=True,
    )

    out: dict[str, ShardedCooMatrix] = {}
    tables_names = preprocessed_inputs.lhs_row_pointers.keys()
    for table_name in tables_names:
        shard_ends = preprocessed_inputs.lhs_row_pointers[table_name]
        shard_starts = np.concatenate(
            [np.asarray([0]), _round_up_to_multiple(shard_ends[:-1], 8)]
        )
        out[table_name] = ShardedCooMatrix(
            shard_starts=shard_starts,
            shard_ends=shard_ends,
            col_ids=preprocessed_inputs.lhs_embedding_ids[table_name],
            row_ids=preprocessed_inputs.lhs_sample_ids[table_name],
            values=preprocessed_inputs.lhs_gains[table_name],
        )

    return out, stats
