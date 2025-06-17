"""Conversion utilities for Keras DistributedEmbeddingConfig to JAX."""

import inspect
import math
import random as python_random
from typing import Any, Callable

import jax
import jax.numpy as jnp
import keras
import numpy as np
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec

from keras_rs.src import types
from keras_rs.src.layers.embedding import distributed_embedding_config as config
from keras_rs.src.types import Nested


class WrappedKerasInitializer(jax.nn.initializers.Initializer):
    """Wraps a Keras initializer for use in JAX."""

    def __init__(self, initializer: keras.initializers.Initializer):
        if isinstance(initializer, str):
            initializer = keras.initializers.get(initializer)
        self.initializer = initializer

    def key(self) -> jax.Array | None:
        """Extract a key from the underlying keras initializer."""
        # All built-in keras initializers have a `seed` attribute.
        # Extract this and turn it into a key for use with JAX.
        if hasattr(self.initializer, "seed"):
            output: jax.Array = keras.src.backend.jax.random.jax_draw_seed(
                self.initializer.seed
            )
            return output
        return None

    def __call__(
        self,
        key: Any,
        shape: Any,
        dtype: Any = jnp.float_,
        out_sharding: Any = None,
    ) -> jax.Array:
        del out_sharding
        # Force use of provided key.  The JAX backend for random initializers
        # forwards the `seed` attribute to the underlying JAX random functions.
        if key is not None and hasattr(self.initializer, "seed"):
            old_seed = self.initializer.seed
            self.initializer.seed = key
            out: jax.Array = self.initializer(shape, dtype)
            self.initializer.seed = old_seed
            return out

        output: jax.Array = self.initializer(shape, dtype)
        return output


# pylint: disable-next=g-classes-have-attributes
class WrappedJaxInitializer(keras.initializers.Initializer):
    """Wraps a JAX initializer for use in Keras.

    Attributes:
        initializer: The wrapped JAX initializer.

    Args:
        initializer: The JAX initializer to wrap.
        seed: Optional Keras seed for use with random JAX initializers.
    """

    def __init__(
        self,
        initializer: jax.nn.initializers.Initializer,
        seed: int | keras.random.SeedGenerator | None = None,
    ):
        self.initializer = initializer
        if seed is None:
            # Consistency with keras.random.make_default_seed().
            seed = python_random.randint(1, int(1e9))
        self.seed = seed

    def key(self) -> jax.Array:
        """Converts the interal seed to a JAX random key."""
        seed = self.seed
        if isinstance(seed, int):
            return jax.random.key(self.seed)
        elif isinstance(seed, keras.random.SeedGenerator):
            return jax.random.key(seed.next())
        elif isinstance(seed, jax.Array):
            return seed
        else:
            raise ValueError(f"Unknown seed {seed} of type {type(seed)}.")

    def __call__(
        self,
        shape: types.Shape,
        dtype: types.DType | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        del kwargs  # Unused.
        return self.initializer(self.key(), shape, dtype)


def keras_to_jax_initializer(
    initializer: str | keras.initializers.Initializer,
) -> jax.nn.initializers.Initializer:
    """Converts a Keras initializer to a JAX initializer.

    Args:
        initializer: Keras initializer to convert.

    Returns:
        A JAX-compatible equivalent initializer.
    """
    if isinstance(initializer, WrappedJaxInitializer):
        return initializer.initializer
    return WrappedKerasInitializer(initializer)


def jax_to_keras_initializer(
    initializer: jax.nn.initializers.Initializer,
) -> keras.initializers.Initializer:
    """Converts a JAX initializer to a Keras initializer.

    Args:
        initializer: JAX initializer to convert.

    Returns:
        An equivalent Keras initializer.
    """
    if isinstance(initializer, WrappedKerasInitializer):
        return initializer.initializer
    return WrappedJaxInitializer(initializer)


def keras_to_jte_learning_rate(
    learning_rate: keras.Variable | float | Callable[..., float],
) -> float | Callable[..., float]:
    """Converts a Keras learning rate to a JAX TPU Embedding learning rate.

    Args:
      learning_rate: Any Keras-compatible learning-rate type.  If a Callable,
          must either take no parameters, or the step size as a single argument.

    Returns:
        A JAX TPU Embedding learning rate.

    Raises:
        ValueError if the learning rate is not supported.
    """

    # Supported keras optimizer general options.
    if isinstance(learning_rate, keras.Variable):
        # Extract the first (and only) element of the variable.
        learning_rate = np.array(learning_rate.value, dtype=float)
        assert learning_rate.size == 1
        lr_float: float = learning_rate.item(0)
        return lr_float
    elif callable(learning_rate):
        # Callable learning rate functions are expected to take a singular step
        # count argument, or no arguments.
        args = inspect.getfullargspec(learning_rate).args
        # If not a function, then it's an object instance with `self` as the
        # first argument.
        num_args = (
            len(args) if inspect.isfunction(learning_rate) else len(args) - 1
        )
        if num_args <= 1:
            return learning_rate
    elif isinstance(learning_rate, float):
        return learning_rate

    raise ValueError(
        f"Unsupported learning rate: {learning_rate} of type"
        f" {type(learning_rate)}."
    )


def jte_to_keras_learning_rate(
    learning_rate: float | Callable[..., float],
) -> float | Callable[..., float]:
    """Converts a JAX TPU Embedding learning rate to a Keras learning rate.

    Args:
        learning_rate: The learning rate value or function.  If a Callable, must
            either take no parameters, or the step size as a single argument.

    Returns:
        A JAX TPU Embedding learning rate.

    Raises:
        ValueError if the learning rate is not supported.
    """
    if callable(learning_rate):
        # Callable learning rate functions are expected to take a singular step
        # count argument, or no arguments.
        args = inspect.getfullargspec(learning_rate).args
        # If not a function, then it's an object instance, with `self` as the
        # first arguments.
        num_args = (
            len(args) if inspect.isfunction(learning_rate) else len(args) - 1
        )
        if num_args <= 1:
            return learning_rate
    elif isinstance(learning_rate, float):
        return learning_rate

    raise ValueError(f"Unknown learning rate {learning_rate}")


def keras_to_jte_optimizer(
    optimizer: keras.optimizers.Optimizer | str,
) -> embedding_spec.OptimizerSpec:
    """Converts a Keras optimizer to a JAX TPU Embedding optimizer.

    Args:
        optimizer: Any Keras-compatible optimizer.

    Returns:
        A JAX TPU Embedding optimizer.
    """
    if isinstance(optimizer, str):
        optimizer = keras.optimizers.get(optimizer)

    # We need to extract the actual internal learning_rate function.
    # Unfortunately, the optimizer.learning_rate attribute tries to be smart,
    # and evaluates the learning rate at the current iteration step, which is
    # not what we want.
    # pylint: disable-next=protected-access
    learning_rate = keras_to_jte_learning_rate(optimizer._learning_rate)

    # Unsupported keras optimizer general options.
    if optimizer.clipnorm is not None:
        raise ValueError("Unsupported optimizer option `clipnorm`.")
    if optimizer.global_clipnorm is not None:
        raise ValueError("Unsupported optimizer option `global_clipnorm`.")
    if optimizer.use_ema:
        raise ValueError("Unsupported optimizer option `use_ema`.")
    if optimizer.loss_scale_factor is not None:
        raise ValueError("Unsupported optimizer option `loss_scale_factor`.")

    # Supported optimizers.
    if isinstance(optimizer, keras.optimizers.SGD):
        if getattr(optimizer, "nesterov", False):
            raise ValueError("Unsupported optimizer option `nesterov`.")
        if getattr(optimizer, "momentum", 0.0) != 0.0:
            raise ValueError("Unsupported optimizer option `momentum`.")
        return embedding_spec.SGDOptimizerSpec(learning_rate=learning_rate)
    elif isinstance(optimizer, keras.optimizers.Adagrad):
        if getattr(optimizer, "epsilon", 1e-7) != 1e-7:
            raise ValueError("Unsupported optimizer option `epsilon`.")
        return embedding_spec.AdagradOptimizerSpec(
            learning_rate=learning_rate,
            initial_accumulator_value=optimizer.initial_accumulator_value,
        )
    elif isinstance(optimizer, keras.optimizers.Adam):
        if getattr(optimizer, "amsgrad", False):
            raise ValueError("Unsupported optimizer option `amsgrad`.")

        return embedding_spec.AdamOptimizerSpec(
            learning_rate=learning_rate,
            beta_1=optimizer.beta_1,
            beta_2=optimizer.beta_2,
            epsilon=optimizer.epsilon,
        )
    elif isinstance(optimizer, keras.optimizers.Ftrl):
        if (
            getattr(optimizer, "l2_shrinkage_regularization_strength", 0.0)
            != 0.0
        ):
            raise ValueError(
                "Unsupported optimizer option "
                "`l2_shrinkage_regularization_strength`."
            )

        return embedding_spec.FTRLOptimizerSpec(
            learning_rate=learning_rate,
            learning_rate_power=optimizer.learning_rate_power,
            l1_regularization_strength=optimizer.l1_regularization_strength,
            l2_regularization_strength=optimizer.l2_regularization_strength,
            beta=optimizer.beta,
            initial_accumulator_value=optimizer.initial_accumulator_value,
        )

    raise ValueError(
        f"Unsupported optimizer type {type(optimizer)}. Optimizer must be "
        f"one of [Adagrad, Adam, Ftrl, SGD]."
    )


def jte_to_keras_optimizer(
    optimizer: embedding_spec.OptimizerSpec,
) -> keras.optimizers.Optimizer:
    """Converts a JAX TPU Embedding optimizer to a Keras optimizer.

    Args:
        optimizer: The JAX TPU Embedding optimizer.

    Returns:
        A corresponding Keras optimizer.
    """
    learning_rate = jte_to_keras_learning_rate(optimizer.learning_rate)
    if isinstance(optimizer, embedding_spec.SGDOptimizerSpec):
        return keras.optimizers.SGD(learning_rate=learning_rate)
    elif isinstance(optimizer, embedding_spec.AdagradOptimizerSpec):
        return keras.optimizers.Adagrad(
            learning_rate=learning_rate,
            initial_accumulator_value=optimizer.initial_accumulator_value,
        )
    elif isinstance(optimizer, embedding_spec.AdamOptimizerSpec):
        return keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=optimizer.beta_1,
            beta_2=optimizer.beta_2,
            epsilon=optimizer.epsilon,
        )
    elif isinstance(optimizer, embedding_spec.FTRLOptimizerSpec):
        if getattr(optimizer, "initial_linear_value", 0.0) != 0.0:
            raise ValueError(
                "Unsupported optimizer option `initial_linear_value`."
            )
        if getattr(optimizer, "multiply_linear_by_learning_rate", False):
            raise ValueError(
                "Unsupported optimizer option "
                "`multiply_linear_by_learning_rate`."
            )
        return keras.optimizers.Ftrl(
            learning_rate=learning_rate,
            learning_rate_power=optimizer.learning_rate_power,
            initial_accumulator_value=optimizer.initial_accumulator_value,
            l1_regularization_strength=optimizer.l1_regularization_strength,
            l2_regularization_strength=optimizer.l2_regularization_strength,
            beta=optimizer.beta,
        )

    raise ValueError(f"Unknown optimizer spec {type(optimizer)}.")


def _keras_to_jte_table_config(
    table_config: config.TableConfig,
) -> embedding_spec.TableSpec:
    # Initializer could be none.  Default to truncated normal.
    initializer = table_config.initializer
    if initializer is None:
        initializer = keras.initializers.TruncatedNormal(
            mean=0.0, stddev=1.0 / math.sqrt(float(table_config.embedding_dim))
        )
    return embedding_spec.TableSpec(
        name=table_config.name,
        vocabulary_size=table_config.vocabulary_size,
        embedding_dim=table_config.embedding_dim,
        initializer=keras_to_jax_initializer(initializer),
        optimizer=keras_to_jte_optimizer(table_config.optimizer),
        combiner=table_config.combiner,
        max_ids_per_partition=table_config.max_ids_per_partition,
        max_unique_ids_per_partition=table_config.max_unique_ids_per_partition,
    )


def keras_to_jte_table_configs(
    table_configs: Nested[config.TableConfig],
) -> Nested[embedding_spec.TableSpec]:
    """Converts Keras RS `TableConfig`s to JAX TPU Embedding `TableSpec`s."""
    return keras.tree.map_structure(
        _keras_to_jte_table_config,
        table_configs,
    )


def _jte_to_keras_table_config(
    table_spec: embedding_spec.TableSpec,
) -> config.TableConfig:
    return config.TableConfig(
        name=table_spec.name,
        vocabulary_size=table_spec.vocabulary_size,
        embedding_dim=table_spec.embedding_dim,
        initializer=jax_to_keras_initializer(table_spec.initializer),
        optimizer=jte_to_keras_optimizer(table_spec.optimizer),
        combiner=table_spec.combiner,
        max_ids_per_partition=table_spec.max_ids_per_partition,
        max_unique_ids_per_partition=table_spec.max_unique_ids_per_partition,
    )


def jte_to_keras_table_configs(
    table_specs: Nested[embedding_spec.TableSpec],
) -> Nested[config.TableConfig]:
    """Converts JAX TPU Embedding `TableSpec`s to Keras RS `TableConfig`s."""
    output: Nested[config.TableConfig] = keras.tree.map_structure(
        _jte_to_keras_table_config,
        table_specs,
    )
    return output


def _keras_to_jte_feature_config(
    feature_config: config.FeatureConfig,
    table_spec_map: dict[str, embedding_spec.TableSpec],
) -> embedding_spec.FeatureSpec:
    table_spec = table_spec_map.get(feature_config.table.name, None)
    if table_spec is None:
        table_spec = _keras_to_jte_table_config(feature_config.table)
        table_spec_map[feature_config.table.name] = table_spec

    return embedding_spec.FeatureSpec(
        name=feature_config.name,
        table_spec=table_spec,
        input_shape=feature_config.input_shape,
        output_shape=feature_config.output_shape,
    )


def keras_to_jte_feature_configs(
    feature_configs: Nested[config.FeatureConfig],
) -> Nested[embedding_spec.FeatureSpec]:
    """Converts Keras RS `FeatureConfig`s to JAX TPU Embedding `FeatureSpec`s.

    Args:
        feature_configs: Keras RS feature configurations.

    Returns:
        JAX TPU Embedding feature specifications.
    """
    table_spec_map: dict[str, embedding_spec.TableSpec] = {}
    return keras.tree.map_structure(
        lambda feature_config: _keras_to_jte_feature_config(
            feature_config, table_spec_map
        ),
        feature_configs,
    )


def _jte_to_keras_feature_config(
    feature_spec: embedding_spec.FeatureSpec,
    table_config_map: dict[str, config.TableConfig],
) -> config.FeatureConfig:
    table_config = table_config_map.get(feature_spec.table_spec.name, None)
    if table_config is None:
        table_config = _jte_to_keras_table_config(feature_spec.table_spec)
        table_config_map[feature_spec.table_spec.name] = table_config

    return config.FeatureConfig(
        name=feature_spec.name,
        table=table_config,
        input_shape=feature_spec.input_shape,
        output_shape=feature_spec.output_shape,
    )


def jte_to_keras_feature_configs(
    feature_specs: Nested[embedding_spec.FeatureSpec],
) -> Nested[config.FeatureConfig]:
    """Converts JAX TPU Embedding `FeatureSpec`s to Keras RS `FeatureConfig`s.

    Args:
        feature_specs: JAX TPU Embedding feature specifications.

    Returns:
        Keras RS feature configurations.
    """
    table_config_map: dict[str, config.TableConfig] = {}
    output: Nested[config.FeatureConfig] = keras.tree.map_structure(
        lambda feature_spec: _jte_to_keras_feature_config(
            feature_spec, table_config_map
        ),
        feature_specs,
    )
    return output
