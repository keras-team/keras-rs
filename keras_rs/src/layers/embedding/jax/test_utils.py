"""JAX-specific test utilities for embedding layers."""

from typing import Mapping, Optional, Sequence, Tuple, TypeVar, Union

import jax
import numpy as np
import tree
from jax import numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec

from keras_rs.src.layers.embedding.jax import embedding_utils

T = TypeVar("T")
Nested = Union[T, Sequence[T], Mapping[str, T]]
ArrayLike = Union[jax.Array, np.ndarray]
Shape = Tuple[int, ...]
FeatureSpec = embedding_spec.FeatureSpec
TableSpec = embedding_spec.TableSpec
FeatureSamples = embedding_utils.FeatureSamples


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def create_table_spec(
    name: str,
    vocabulary_size: int,
    embedding_dim: int,
    max_ids_per_partition: int = 0,
    max_unique_ids_per_partition: int = 0,
    initializer: Optional[jax.nn.initializers.Initializer] = None,
    optimizer: Optional[embedding_spec.OptimizerSpec] = None,
    combiner: str = "sum",
) -> TableSpec:
    """Creates a TableSpec with appropriate defaults."""
    max_ids_per_partition = (
        max_ids_per_partition if max_ids_per_partition > 0 else vocabulary_size
    )
    max_unique_ids_per_partition = (
        max_unique_ids_per_partition
        if max_unique_ids_per_partition > 0
        else max_ids_per_partition
    )

    initializer = initializer or jax.nn.initializers.uniform()
    optimizer = optimizer or embedding_spec.SGDOptimizerSpec()

    return TableSpec(
        name=name,
        vocabulary_size=vocabulary_size,
        embedding_dim=embedding_dim,
        initializer=initializer,
        optimizer=optimizer,
        combiner=combiner,
        max_ids_per_partition=max_ids_per_partition,
        max_unique_ids_per_partition=max_unique_ids_per_partition,
    )


def create_feature_spec(
    name: str, table_spec: TableSpec, batch_size: int, sample_size: int
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        table_spec=table_spec,
        input_shape=(batch_size, sample_size),
        output_shape=(
            batch_size,
            table_spec.embedding_dim,
        ),
    )


def _default_stacked_table_spec(
    table_spec: TableSpec, num_shards: int
) -> embedding_spec.StackedTableSpec:
    return embedding_spec.StackedTableSpec(
        stack_name=table_spec.name,
        stack_vocab_size=_round_up_to_multiple(
            table_spec.vocabulary_size, 8 * num_shards
        ),
        stack_embedding_dim=_round_up_to_multiple(table_spec.embedding_dim, 8),
        optimizer=table_spec.optimizer,
        combiner=table_spec.combiner,
        total_sample_count=0,
        max_ids_per_partition=table_spec.max_ids_per_partition,
        max_unique_ids_per_partition=table_spec.max_unique_ids_per_partition,
    )


def _get_stacked_table_spec(
    table_spec: TableSpec, num_shards: int = 1
) -> embedding_spec.StackedTableSpec:
    return table_spec.stacked_table_spec or _default_stacked_table_spec(
        table_spec, num_shards
    )


def create_tables(
    table_specs: Nested[TableSpec],
    keys: Optional[Nested[ArrayLike]] = None,
) -> Nested[ArrayLike]:
    """Creates and initializes embedding tables.

    Args:
      table_specs: A nested collection of table specifications.
      keys: A nested collection of keys to use for initialization.

    Returns:
      The set of initialized tables.s
    """
    if keys is None:
        keys = jax.random.key(0)

    if tree.is_nested(table_specs) and not tree.is_nested(keys):
        tree_size = len(tree.flatten(table_specs))
        keys = jnp.unstack(jax.random.split(keys, tree_size))
        keys = tree.unflatten_as(table_specs, keys)

    # Initialize tables.
    return tree.map_structure(
        lambda table_spec, key: table_spec.initializer(
            key,
            (table_spec.vocabulary_size, table_spec.embedding_dim),
            dtype=jnp.float32,
        ),
        table_specs,
        keys,
    )


def create_table_and_slot_variables(
    table_specs: Nested[TableSpec],
    keys: Optional[Nested[ArrayLike]] = None,
) -> Nested[ArrayLike]:
    """Creates and initializes embedding tables and slot variables.

    Args:
      table_specs: A nested collection of table specifications.
      keys: A nested collection of keys to use for initialization.

    Returns:
      The set of initialized tables and gradient slot variables.
    """
    if keys is None:
        keys = jax.random.key(0)

    if tree.is_nested(table_specs) and not tree.is_nested(keys):
        tree_size = len(tree.flatten(table_specs))
        keys = [key for key in jax.random.split(keys, tree_size)]
        keys = tree.unflatten_as(table_specs, keys)

    def _create_table_and_slot_variables(
        table_spec: TableSpec,
        key: ArrayLike,
    ):
        slot_initializers = table_spec.optimizer.slot_variables_initializers()
        num_slot_variables = len(tree.flatten(slot_initializers))
        slot_keys = jnp.unstack(jax.random.split(key, num_slot_variables))
        slot_keys = tree.unflatten_as(slot_initializers, slot_keys)
        table_shape = (table_spec.vocabulary_size, table_spec.embedding_dim)
        table = table_spec.initializer(key, table_shape, dtype=jnp.float32)
        slot_variables = tree.map_structure(
            lambda initializer, key: initializer(
                key, table_shape, dtype=jnp.float32
            ),
            slot_initializers,
            slot_keys,
        )
        return (table, slot_variables)

    # Initialize tables.
    return tree.map_structure(
        _create_table_and_slot_variables,
        table_specs,
        keys,
    )


def create_feature_samples(
    feature_specs: Nested[FeatureSpec],
    max_samples: Nested[int] = 16,
    ragged: bool = True,
    keys: Optional[Nested[int]] = None,
    sample_weight_initializer: Optional[
        Nested[jax.nn.initializers.Initializer]
    ] = None,
) -> Nested[FeatureSamples]:
    """Creates random feature samples for embedding lookup testing.

    Args:
      feature_specs: A nested collection of feature specifications.
      max_samples: The maximum number of samples to generate per feature.
      ragged: Whether to generate ragged or dense samples.
      keys: A nested collection of keys to use for initialization.
      sample_weight_initializer: The initializer to use for sample weights.

    Returns:
      The collection of generated feature samples.
    """
    sample_weight_initializer = (
        sample_weight_initializer or jax.nn.initializers.uniform()
    )

    keys = keys or jax.random.key(0)

    if tree.is_nested(feature_specs):
        if not tree.is_nested(keys):
            tree_size = len(tree.flatten(feature_specs))
            keys = jnp.unstack(jax.random.split(keys, tree_size))
            keys = tree.unflatten_as(feature_specs, keys)

        # Extend properties to the entire tree.
        if not tree.is_nested(max_samples):
            max_samples = tree.map_structure(
                lambda _: max_samples, feature_specs
            )

        if not tree.is_nested(sample_weight_initializer):
            sample_weight_initializer = tree.map_structure(
                lambda _: sample_weight_initializer, feature_specs
            )

    def _create_samples(
        feature_spec: FeatureSpec,
        feature_max_samples: int,
        key: ArrayLike,
        weight_initializer: jax.nn.initializers.Initializer,
    ) -> FeatureSamples:
        batch_size = feature_spec.input_shape[0]
        sample_size = feature_spec.input_shape[1]
        vocabulary_size = feature_spec.table_spec.vocabulary_size

        if ragged:
            # indptr
            counts = jax.random.randint(
                key,
                minval=0,
                maxval=feature_max_samples,
                shape=(batch_size,),
                dtype=jnp.int32,
            )
            samples = np.empty(batch_size, dtype=np.ndarray)
            weights = np.empty(batch_size, dtype=np.ndarray)
            keys = jax.random.split(key, batch_size)
            for i in range(batch_size):
                skey, dkey = jax.random.split(keys[i])
                samples[i] = np.asarray(
                    jax.random.randint(
                        skey,
                        minval=0,
                        maxval=vocabulary_size,
                        shape=counts[i].item(0),
                        dtype=jnp.int32,
                    )
                )
                weights[i] = np.asarray(
                    weight_initializer(dkey, (counts[i],), jnp.float32)
                )

            return FeatureSamples(
                samples,
                weights,
            )
        else:
            skey, dkey = jax.random.split(key)
            samples = jax.random.randint(
                skey,
                minval=0,
                maxval=vocabulary_size,
                shape=(batch_size, sample_size),
                dtype=jnp.int32,
            )
            weights = weight_initializer(
                dkey, (batch_size, sample_size), dtype=jnp.float32
            )
            # Pad with zeros if beyond feature_max_samples.
            idx = jnp.arange(sample_size)
            samples = jnp.where(idx < feature_max_samples, samples, 0)
            weights = jnp.where(idx < feature_max_samples, weights, 0)
            return FeatureSamples(samples, weights)

    return tree.map_structure(
        _create_samples,
        feature_specs,
        max_samples,
        keys,
        sample_weight_initializer,
    )


def stack_shard_and_put_tables(
    table_specs: Nested[TableSpec],
    tables: Nested[jax.Array],
    num_shards: int,
    sharding: jax.sharding.Sharding,
) -> dict[str, Nested[jax.Array]]:
    sharded_tables = embedding_utils.stack_and_shard_tables(
        table_specs, tables, num_shards
    )
    return jax.device_put(
        jax.tree.map(
            # Flatten shard dimension to allow auto-sharding to split the array.
            lambda table: table.reshape((-1, table.shape[-1])),
            sharded_tables,
        ),
        sharding,
    )


def get_unshard_and_unstack_tables(
    table_specs: Nested[TableSpec],
    sharded_tables: Nested[jax.Array],
    num_shards: int,
) -> Nested[jax.Array]:
    sharded_tables = jax.device_get(sharded_tables)
    return embedding_utils.unshard_and_unstack_tables(
        table_specs, sharded_tables, num_shards
    )


def _compute_expected_lookup(
    samples: FeatureSamples,
    table: ArrayLike,
) -> ArrayLike:
    """Manually does a Sparse-Dense multiplication for embedding lookup."""
    batch_size = len(samples.tokens)
    out = jnp.zeros(shape=(batch_size, table.shape[1]), dtype=table.dtype)
    for i in range(len(samples.tokens)):
        out = out.at[i, :].add(samples.weights[i] @ table[samples.tokens[i], :])

    return out


def compute_expected_lookup(
    feature_specs: Nested[FeatureSpec],
    feature_samples: Nested[FeatureSamples],
    table_specs: Nested[TableSpec],
    tables: Nested[jax.Array],
) -> Nested[jax.Array]:
    """Computes the expected output of an embedding lookup.

    Args:
      feature_specs: A nested collection of feature specifications.
      feature_samples: Corresponding collection of feature samples.
      table_specs: A nested collection of embedding table specifications.
      tables: Corresponding collection of embedding table values.

    Returns:
      The expected output of the embedding lookup.
    """
    tree.assert_same_structure(table_specs, tables)

    # Collect table information.
    table_map = {
        table_spec.name: table
        for table_spec, table in zip(
            tree.flatten(table_specs), tree.flatten(tables)
        )
    }

    return tree.map_structure_up_to(
        feature_specs,
        lambda feature_spec, samples: _compute_expected_lookup(
            samples, table_map[feature_spec.table_spec.name]
        ),
        feature_specs,
        feature_samples,
    )


def _compute_expected_lookup_grad(
    samples: FeatureSamples,
    vocabulary_size: int,
    activation_gradients: ArrayLike,
) -> ArrayLike:
    """Computes the expected gradient of an embedding lookup."""
    # Convert to COO.
    batch_size = activation_gradients.shape[0]
    embedding_dim = activation_gradients.shape[1]
    sample_lengths = jnp.array([len(sample) for sample in samples.tokens])
    rows = jnp.repeat(jnp.arange(batch_size), sample_lengths)
    cols = jnp.concatenate(np.unstack(samples.tokens))
    vals = jnp.concatenate(np.unstack(samples.weights)).reshape(-1, 1)

    # Compute: grad = samples^T * activation_gradients.
    grad = jnp.zeros(shape=(vocabulary_size, embedding_dim))
    grad = grad.at[cols, :].add(
        vals * activation_gradients[rows, :],
    )
    return grad


def compute_expected_lookup_grad(
    feature_specs: Nested[FeatureSpec],
    feature_samples: Nested[FeatureSamples],
    activation_gradients: Nested[jax.Array],
    table_specs: Nested[TableSpec],
) -> Tuple[None, Nested[jax.Array]]:
    """Computes the expected gradient of an embedding lookup.

    Args:
      feature_specs: A nested collection of feature specifications.
      feature_samples: Corresponding collection of feature samples.
      activation_gradients: The gradient of the embedding layer output for
        back-propagation.
      table_specs: The nested collection of embedding table specifications.

    Returns:
      The gradients for the layer w.r.t. the feature samples and tables.
    """
    tree.assert_same_structure(feature_specs, activation_gradients)

    per_feature_table_grads = tree.map_structure_up_to(
        feature_specs,
        lambda feature_spec, samples, grad: _compute_expected_lookup_grad(
            samples, feature_spec.table_spec.vocabulary_size, grad
        ),
        feature_specs,
        feature_samples,
        activation_gradients,
    )

    # Accumulate across features to determine a per-table gradient.
    per_feature_table_grads = tree.flatten_up_to(
        feature_specs, per_feature_table_grads
    )
    flat_feature_specs = tree.flatten(feature_specs)
    table_grads = {}
    for feature_spec, per_feature_table_grad in zip(
        flat_feature_specs, per_feature_table_grads
    ):
        table_name = feature_spec.table_spec.name
        if table_name not in table_grads:
            table_grads[table_name] = per_feature_table_grad
        else:
            table_grads[table_name] = (
                table_grads[table_name] + per_feature_table_grad
            )

    table_grads = tree.map_structure(
        lambda table_spec: table_grads[table_spec.name],
        table_specs,
    )

    # No gradient w.r.t. feature samples.
    return None, table_grads


def _update_table_and_slot_variables(
    table_spec: TableSpec,
    grad: jax.Array,
    table_and_slot_variables: Nested[jax.Array],
) -> Tuple[
    jax.Array,
    Union[embedding_spec.SGDSlotVariables, embedding_spec.AdagradSlotVariables],
]:
    """Updates a table and its slot variables based on the gradient."""
    table = table_and_slot_variables[0]
    optimizer = table_spec.optimizer

    # Adagrad, update and apply gradient accumulator.
    if isinstance(optimizer, embedding_spec.AdagradOptimizerSpec):
        accumulator = table_and_slot_variables[1][0]
        accumulator = accumulator + grad * grad
        learning_rate = optimizer.get_learning_rate(0) / jnp.sqrt(accumulator)
        return (
            table - learning_rate * grad,
            embedding_spec.AdagradSlotVariables(accumulator=accumulator),
        )

    # SGD
    return (
        table - optimizer.get_learning_rate(0) * grad,
        embedding_spec.SGDSlotVariables(),
    )


def compute_expected_updates(
    feature_specs: Nested[FeatureSpec],
    feature_samples: Nested[FeatureSamples],
    activation_gradients: Nested[jax.Array],
    table_specs: Nested[TableSpec],
    table_and_slot_variables: Nested[jax.Array],
) -> Nested[jax.Array]:
    """Computes the expected updates for a given embedding lookup.

    Args:
      feature_specs: A nested collection of feature specifications.
      feature_samples: Corresponding collection of feature samples.
      activation_gradients: The gradient of the embedding layer output for
        back-propagation.
      table_specs: The nested collection of embedding table specifications.
      table_and_slot_variables: The nested collection of embedding table and
        gradient slot variable values.

    Returns:
      The expected updated values for the embedding tables and gradient slot
      variables.
    """
    _, table_grads = compute_expected_lookup_grad(
        feature_specs, feature_samples, activation_gradients, table_specs
    )

    # Apply updates per table.
    return tree.map_structure_up_to(
        table_specs,
        _update_table_and_slot_variables,
        table_specs,
        table_grads,
        table_and_slot_variables,
    )
