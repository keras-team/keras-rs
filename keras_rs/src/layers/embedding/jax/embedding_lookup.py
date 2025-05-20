"""Defines a differentiable `embedding_lookup` function.

Implementation details for use in JAX models.
"""

import functools
from typing import Any, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import jax
import numpy as np
from jax.experimental import layout
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.utils import utils as jte_utils

from keras_rs.src.layers.embedding.jax import embedding_utils

ShardedCooMatrix = embedding_utils.ShardedCooMatrix
shard_map = jax.experimental.shard_map.shard_map

T = TypeVar("T")
Nested = Union[T, Sequence[T], Mapping[str, T]]
ArrayLike = Union[jax.Array, np.ndarray[Any, Any]]
JaxLayout = Union[jax.sharding.NamedSharding, layout.Layout]


class EmbeddingLookupConfiguration:
    """Feature, mesh and sharding configrmation for lookups."""

    mesh: jax.sharding.Mesh
    feature_specs: embedding.Nested[embedding_spec.FeatureSpec]
    table_sharding_strategy: str
    num_sc_per_device: int
    samples_partition: jax.sharding.PartitionSpec
    table_partition: jax.sharding.PartitionSpec
    table_layout: JaxLayout

    def __init__(
        self,
        feature_specs: embedding.Nested[embedding_spec.FeatureSpec],
        mesh: Optional[jax.sharding.Mesh] = None,
        table_sharding_strategy: str = "MOD",
        num_sc_per_device: Optional[int] = None,
        sharding_axis: str = "sparsecore_sharding",
        samples_partition: Optional[jax.sharding.PartitionSpec] = None,
        samples_layout: Optional[JaxLayout] = None,
        table_partition: Optional[jax.sharding.PartitionSpec] = None,
        table_layout: Optional[JaxLayout] = None,
    ):
        self.mesh = mesh or jax.sharding.Mesh(jax.devices(), sharding_axis)
        self.feature_specs = feature_specs
        self.table_sharding_strategy = table_sharding_strategy
        self.num_sc_per_device = (
            num_sc_per_device
            if num_sc_per_device is not None
            else jte_utils.num_sparsecores_per_device()
        )
        self.samples_partition = (
            samples_partition
            or jax.sharding.PartitionSpec(
                sharding_axis  # type: ignore[no-untyped-call]
            )
        )
        self.samples_layout = samples_layout or jax.sharding.NamedSharding(
            self.mesh, self.samples_partition
        )
        self.table_partition = table_partition or jax.sharding.PartitionSpec(
            sharding_axis,
            None,  # type: ignore[no-untyped-call]
        )
        self.table_layout = table_layout or jax.sharding.NamedSharding(
            self.mesh, self.table_partition
        )


# Embedding lookup function with custom gradient.
@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def embedding_lookup(
    config: EmbeddingLookupConfiguration,
    lookups: Mapping[str, ShardedCooMatrix],
    tables: Nested[jax.Array],
    step: Optional[jax.Array] = None,
) -> Nested[jax.Array]:
    """Embedding lookup function with custom gradient.

    Args:
      config: Embedding lookup configuration.
      lookups: Embedding lookup stacked/sharded inputs.
      tables: Embedding lookup stacked/sharded tables.
      step: Current training step number.
    """
    del step  # Only used in backward pass.

    # Decompose COO matrices.
    row_pointers_raw = {}
    embedding_ids_raw = {}
    sample_ids_raw = {}
    gains_raw = {}
    for table_name, coo in lookups.items():
        row_pointers_raw[table_name] = coo.shard_ends
        embedding_ids_raw[table_name] = coo.col_ids
        sample_ids_raw[table_name] = coo.row_ids
        gains_raw[table_name] = coo.values

    sparse_dense_matmul_input = embedding.SparseDenseMatmulInput(
        row_pointers_raw,
        embedding_ids_raw,
        sample_ids_raw,
        gains_raw,
    )

    pd = config.samples_partition
    pt = config.table_partition
    sharded_matmul = jax.jit(
        shard_map(
            functools.partial(
                embedding.tpu_sparse_dense_matmul,
                global_device_count=config.mesh.shape[pd[0]],
                feature_specs=config.feature_specs,
                sharding_strategy=config.table_sharding_strategy,
            ),
            mesh=config.mesh,
            in_specs=(pd, pt),
            out_specs=pd,
            check_rep=False,
        ),
    )

    activations: Nested[jax.Array] = sharded_matmul(
        sparse_dense_matmul_input,
        tables,
    )

    return activations


def embedding_lookup_fwd(
    config: EmbeddingLookupConfiguration,
    lookups: Mapping[str, ShardedCooMatrix],
    table: Nested[jax.Array],
    step: Optional[jax.Array] = None,
) -> Tuple[
    Nested[jax.Array],
    Tuple[Nested[ShardedCooMatrix], Nested[jax.Array], Optional[jax.Array]],
]:
    """Forward pass for embedding lookup."""
    return embedding_lookup(config, lookups, table, step), (
        lookups,
        table,
        step,
    )


def embedding_lookup_bwd(
    config: EmbeddingLookupConfiguration,
    res: Tuple[
        Mapping[str, ShardedCooMatrix],  # Lookups.
        Mapping[str, Nested[jax.Array]],  # Tables.
        Optional[jax.Array],  # Step.
    ],
    gradients: Nested[jax.Array],
) -> Tuple[None, Nested[jax.Array], Optional[jax.Array]]:
    """Backward pass for embedding lookup.

    Args:
      config: Embedding lookup configuration.
      res: Tuple of embedding lookup (inputs, tables, step).
      gradients: Embedding lookup gradients.

    Returns:
      A tuple of gradients (None, table_grads, step + 1).
    """
    lookups, tables, step = res

    # Decompose COO matrices.
    row_pointers_raw = {}
    embedding_ids_raw = {}
    sample_ids_raw = {}
    gains_raw = {}
    for table_name, coo in lookups.items():
        row_pointers_raw[table_name] = coo.shard_ends
        embedding_ids_raw[table_name] = coo.col_ids
        sample_ids_raw[table_name] = coo.row_ids
        gains_raw[table_name] = coo.values

    sparse_dense_matmul_input = embedding.SparseDenseMatmulInput(
        row_pointers_raw,
        embedding_ids_raw,
        sample_ids_raw,
        gains_raw,
    )

    pt = config.table_partition
    pd = config.samples_partition
    # Replicate step count.
    preplicate = jax.sharding.PartitionSpec()  # type: ignore[no-untyped-call]

    def grad_func(
        gradients: Nested[jax.Array],
        sparse_input: Nested[jax.Array],
        tables: Nested[jax.Array],
        step: Optional[jax.Array],
    ) -> Nested[jax.Array]:
        output: Nested[jax.Array] = embedding.tpu_sparse_dense_matmul_grad(
            gradients,
            sparse_input,
            tables,
            feature_specs=config.feature_specs,
            sharding_strategy=config.table_sharding_strategy,
            step=step,
        )
        return output

    # activation_layout = jax.sharding.NamedSharding(config.mesh, pd)
    # step_layout = jax.sharding.NamedSharding(config.mesh, preplicate)
    sharded_matmul_grad = jax.jit(
        shard_map(
            grad_func,
            mesh=config.mesh,
            in_specs=(pd, pd, pt, preplicate),
            out_specs=pt,
            check_rep=False,
        ),
        #   in_shardings=(
        #       activation_layout,
        #       config.samples_layout,
        #       config.table_layout,
        #       step_layout,
        #   ),
        # out_shardings=(config.table_layout),
    )

    table_grads = sharded_matmul_grad(
        gradients,
        sparse_dense_matmul_input,
        tables,
        step,
    )

    # tpu_sparse_dense_matmul_grad returns a general Mapping (usually a dict).
    # It may not be the same type as the embedding table (e.g. FrozenDict).
    # Here we use flatten / unflatten to ensure the types are the same.
    table_grads = jax.tree.unflatten(
        jax.tree.structure(tables), jax.tree.leaves(table_grads)
    )

    return (
        None,
        table_grads,
        step + 1 if step is not None else None,  # Incremented step count.
    )


embedding_lookup.defvjp(embedding_lookup_fwd, embedding_lookup_bwd)
