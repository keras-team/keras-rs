"""Useful utilities for tests."""

import typing
from typing import Any, Callable, Mapping, Sequence, TypeAlias

import keras
import numpy as np

from keras_rs.src.layers.embedding.distributed_embedding_config import (
    FeatureConfig,
)
from keras_rs.src.layers.embedding.distributed_embedding_config import (
    TableConfig,
)
from keras_rs.src.types import Nested

RandomSeed: TypeAlias = int | keras.random.SeedGenerator
AnyNdArray: TypeAlias = np.ndarray[Any, Any]


def _make_rng(seed: RandomSeed | None) -> keras.random.SeedGenerator:
    """Make a seed generator for use in generating random configurations."""
    if seed is None:
        seed = keras.random.SeedGenerator()
    elif isinstance(seed, int):
        seed = keras.random.SeedGenerator(seed)
    return seed


def create_random_table_configs(
    count: int = 3,
    max_vocabulary_size: int = 1024,
    max_embedding_dim: int = 128,
    initializer: str | keras.initializers.Initializer | None = None,
    optimizer: str | keras.optimizers.Optimizer = "sgd",
    combiner: str = "mean",
    placement: str = "auto",
    max_ids_per_partition: int = 256,
    max_unique_ids_per_partition: int = 256,
    seed: RandomSeed | None = None,
    name_prefix: str = "table",
) -> list[TableConfig]:
    """Creates a list of random TableConfigs for use in tests.

    Args:
        count: The number of tables.
        max_vocabulary_size: Maximum size of each table's vocabulary (number of
            rows).
        max_embedding_dim: Maximum size of each table's embedding dimension
            (width).
        initializer: The initializer for the embedding weights.
        optimizer: The optimizer for the table.
        combiner: Specifies how to reduce if there are multiple entries in a
            single row.
        placement: Where to place the embedding table.
        max_ids_per_partition: The maximum number of ids per pertition for the
            table.
        max_unique_ids_per_partition: The maximum number of unique ids per
            partition for the type.
        seed: Random seed for generating table configurations.
        name_prefix: Prefix for generated table names.  The overall table name
              will be `{name_prefix}:{i}` if count > 1, otherwise simply
              `{name_prefix}`.

    Returns:
        List of generated table configurations.
    """
    # Convert to a SeedGenerator to get different tables.
    seed = _make_rng(seed)
    return [
        TableConfig(
            name=f"{name_prefix}:{i}" if count > 1 else name_prefix,
            vocabulary_size=keras.random.randint(
                shape=tuple(), minval=1, maxval=max_vocabulary_size, seed=seed
            ).item(0),
            embedding_dim=keras.random.randint(
                shape=tuple(), minval=1, maxval=max_embedding_dim, seed=seed
            ).item(0),
            initializer=initializer,
            optimizer=optimizer,
            combiner=combiner,
            placement=placement,
            max_ids_per_partition=max_ids_per_partition,
            max_unique_ids_per_partition=max_unique_ids_per_partition,
        )
        for i in range(count)
    ]


def create_random_feature_configs(
    table_configs: Sequence[TableConfig] | None = None,
    max_features_per_table: int = 3,
    batch_size: int = 16,
    seed: RandomSeed | None = None,
    name_prefix: str = "feature",
) -> list[FeatureConfig]:
    """Creates a list of random FeatureConfigs for tests.

    Args:
        table_configs: Table configurations for which to generate features.  If
            `None`, generates a set of random tables.
        max_features_per_table: Maximum number of features to generate per
            table.  At least one feature will be generated regardless.
        batch_size: Input batch size for the feature.
        seed: Random seed for generating features.
        name_prefix: Prefix for feature name.  The overall feature name will be
            `{table.name}:{name_prefix}:{feature_index}` if
            `max_features_per_table > 1`, otherwise
            `{table.name}:{name_prefix}`.

    Returns:
        List of generated feature configurations.
    """
    # Convert to a SeedGenerator to get different features.
    seed = _make_rng(seed)

    table_configs = table_configs or create_random_table_configs(seed=seed)
    features = []
    for table in table_configs:
        num_features = keras.random.randint(
            shape=tuple(), minval=1, maxval=max_features_per_table, seed=seed
        ).item(0)
        for i in range(num_features):
            feature_prefix = f"{table.name}:{name_prefix}"
            features.append(
                FeatureConfig(
                    name=f"{feature_prefix}:{i}"
                    if max_features_per_table > 1
                    else feature_prefix,
                    table=table,
                    input_shape=(
                        batch_size,
                        None,  # Ragged?  JTE never reads the 2nd dim size.
                    ),
                    output_shape=(batch_size, table.embedding_dim),
                )
            )

    return features


def create_random_samples(
    feature_configs: Nested[FeatureConfig],
    ragged: bool = True,
    max_ids_per_sample: int = 20,
    seed: RandomSeed | None = None,
) -> tuple[Nested[AnyNdArray], Nested[AnyNdArray]]:
    """Creates random feature samples.

    Args:
        feature_configs: Nested set of features for which to generate samples.
        ragged: Whether to generate ragged or dense samples.
        max_ids_per_sample: Maximum number of IDs per sample row.
        seed: Initial seed for generating samples.

    Returns:
        A tuple of generated (token IDs, weights) with the same structure as the
        input samples.  The format of the outputs are lists of lists for
        cross-backend compatibility.
    """
    seed = _make_rng(seed)

    def _generate_samples(
        feature_config: FeatureConfig,
    ) -> tuple[AnyNdArray, AnyNdArray]:
        batch_size = typing.cast(int, feature_config.input_shape[0])
        vocabulary_size = feature_config.table.vocabulary_size
        counts = keras.random.randint(
            shape=(batch_size,), minval=0, maxval=max_ids_per_sample, seed=seed
        )
        sample_ids = []
        sample_weights = []
        for i in range(batch_size):
            if ragged:
                sample_ids.append(
                    np.asarray(
                        keras.random.randint(
                            shape=(counts[i],),
                            minval=0,
                            maxval=vocabulary_size,
                            seed=seed,
                        )
                    )
                )
                sample_weights.append(
                    np.asarray(
                        keras.random.uniform(
                            shape=(counts[i],),
                            minval=0,
                            maxval=1,
                            seed=seed,
                            dtype="float32",
                        )
                    )
                )
            else:
                ids = keras.random.randint(
                    shape=(max_ids_per_sample,),
                    minval=0,
                    maxval=vocabulary_size,
                    seed=seed,
                )
                weights = keras.random.uniform(
                    shape=(max_ids_per_sample,),
                    minval=0,
                    maxval=1,
                    seed=seed,
                    dtype="float32",
                )
                # Mask-out tail of dense samples.
                idx = keras.ops.arange(max_ids_per_sample)
                ids = keras.ops.where(idx <= counts[i], ids, 0)
                weights = keras.ops.where(idx <= counts[i], weights, 0)
                sample_ids.append(np.asarray(ids))
                sample_weights.append(np.asarray(weights))

        if ragged:
            output_sample_ids = np.array(sample_ids, dtype=np.ndarray)
            output_sample_weights = np.array(sample_weights, dtype=np.ndarray)
            return output_sample_ids, output_sample_weights

        output_sample_ids = np.array(sample_ids, dtype=np.int32)
        output_sample_weights = np.array(sample_weights, dtype=np.float32)
        return output_sample_ids, output_sample_weights

    feature_ids = []
    feature_weights = []
    for feature_config in keras.tree.flatten(feature_configs):
        ids, weights = _generate_samples(feature_config)
        feature_ids.append(ids)
        feature_weights.append(weights)

    feature_ids = keras.tree.pack_sequence_as(feature_configs, feature_ids)
    feature_weights = keras.tree.pack_sequence_as(
        feature_configs, feature_weights
    )

    return feature_ids, feature_weights


def _compute_expected_lookup(
    sample_ids: AnyNdArray,
    sample_weights: AnyNdArray,
    table: AnyNdArray,
    combiner: str,
) -> AnyNdArray:
    """Manually does a Sparse-Dense multiplication for embedding lookup."""
    batch_size = len(sample_ids)
    out = np.zeros(shape=(batch_size, table.shape[1]), dtype=table.dtype)
    for i in range(batch_size):
        weights = np.asarray(sample_weights[i], dtype=float)
        if combiner == "mean":
            weights = weights / np.sum(weights)
        elif combiner == "sqrtn":
            weights = weights / np.sqrt(np.sum(np.square(weights)))

        np.add.at(
            out,
            np.s_[i, :],  # type: ignore[arg-type]
            weights @ table[sample_ids[i], :],
        )

    return out


def compute_expected_lookup(
    feature_configs: Nested[FeatureConfig],
    tables: Mapping[str, AnyNdArray],
    feature_sample_ids: Nested[AnyNdArray],
    feature_sample_weights: Nested[AnyNdArray],
) -> Nested[AnyNdArray]:
    """Manually compute the expected embedding feature lookup.

    Args:
        feature_configs: Feature configurations.
        tables: Mapping of table name -> table.
        feature_sample_ids: Ragged or dense input sample IDs.
        feature_sample_weights: Ragged or dense input sample weights.

    Returns:
        The expected embedding feature lookups.
    """

    def do_lookup(
        feature_config: FeatureConfig,
        sample_ids: AnyNdArray,
        sample_weights: AnyNdArray,
    ) -> AnyNdArray:
        return _compute_expected_lookup(
            sample_ids,
            sample_weights,
            tables[feature_config.table.name],
            feature_config.table.combiner,
        )

    output: Nested[AnyNdArray] = keras.tree.map_structure_up_to(
        feature_configs,
        do_lookup,
        feature_configs,
        feature_sample_ids,
        feature_sample_weights,
    )
    return output


# pylint: disable-next=g-classes-have-attributes
class RandomInputSampleDataset(keras.utils.PyDataset):
    """Simple dataset iterator that generates random samples.

    Args:
        feature_configs: FeatureConfig instances for the layer.
        tables: Mapping of table name -> table for the embedding tables used to
            generate labels.
        ragged: Controls whether generated inputs are ragged or dense.
        max_ids_per_sample: Maximum number of IDs per sample row.
        num_batches: Number of batches to generate.
        seed: Initial seed for generating samples.
        preprocessor: optional callable preprocessor to apply before outputting
            the data.  This may, for example, apply padding to ensure a
            consistent input size, or may apply additional transforms to the
            data to prepare it for training.
        **kwargs: Arguments to pass to the base dataset.
    """

    def __init__(
        self,
        feature_configs: Nested[FeatureConfig],
        tables: Mapping[str, AnyNdArray],
        ragged: bool = True,
        max_ids_per_sample: int = 20,
        num_batches: int = 1000,
        seed: int = 0,
        preprocessor: Callable[..., Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._feature_configs = feature_configs
        self._tables = tables
        self._ragged = ragged
        self._max_ids_per_sample = max_ids_per_sample
        self._num_batches = num_batches
        self._sample_seeds = keras.random.randint(
            shape=num_batches, minval=0, maxval=2**31 - 1, seed=seed
        )
        self._preprocessor = preprocessor

    def __len__(self) -> int:
        return self._num_batches

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        sample_ids, sample_weights = create_random_samples(
            self._feature_configs,
            self._ragged,
            self._max_ids_per_sample,
            self._sample_seeds.item(idx),
        )

        expected = compute_expected_lookup(
            self._feature_configs, self._tables, sample_ids, sample_weights
        )

        if self._preprocessor:
            inputs = self._preprocessor(sample_ids, sample_weights)
        else:
            inputs = [sample_ids, sample_weights]

        return inputs, expected
