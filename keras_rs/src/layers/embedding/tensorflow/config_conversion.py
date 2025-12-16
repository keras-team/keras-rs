import collections
from typing import Any, Sequence, TypeAlias

import keras
import tensorflow as tf

from keras_rs.src import types
from keras_rs.src.layers.embedding import distributed_embedding_config

FeatureConfig = distributed_embedding_config.FeatureConfig
TableConfig = distributed_embedding_config.TableConfig

# Placeholder of tf.tpu.experimental.embedding._Optimizer which is not exposed.
TfTpuOptimizer: TypeAlias = Any


OptimizerMapping = collections.namedtuple(
    "OptimizerMapping",
    ["tpu_optimizer_class", "supported_kwargs", "unsupported_kwargs"],
)


OPTIMIZER_MAPPINGS = {
    keras.optimizers.Adagrad: OptimizerMapping(
        tpu_optimizer_class=tf.tpu.experimental.embedding.Adagrad,
        supported_kwargs=["initial_accumulator_value"],
        unsupported_kwargs={"epsilon": 1e-07},
    ),
    keras.optimizers.Adam: OptimizerMapping(
        tpu_optimizer_class=tf.tpu.experimental.embedding.Adam,
        supported_kwargs=["beta_1", "beta_2", "epsilon"],
        unsupported_kwargs={"amsgrad": False},
    ),
    keras.optimizers.Ftrl: OptimizerMapping(
        tpu_optimizer_class=tf.tpu.experimental.embedding.FTRL,
        supported_kwargs=[
            "learning_rate_power",
            "initial_accumulator_value",
            "l1_regularization_strength",
            "l2_regularization_strength",
            "beta",
        ],
        unsupported_kwargs={"l2_shrinkage_regularization_strength": 0.0},
    ),
    keras.optimizers.SGD: OptimizerMapping(
        tpu_optimizer_class=tf.tpu.experimental.embedding.SGD,
        supported_kwargs=[],
        unsupported_kwargs={"momentum": 0.0, "nesterov": False},
    ),
}


# KerasRS to TensorFlow


def keras_to_tf_tpu_configuration(
    feature_configs: types.Nested[FeatureConfig],
    table_stacking: str | Sequence[str] | Sequence[Sequence[str]],
    num_replicas_in_sync: int,
) -> tuple[
    types.Nested[tf.tpu.experimental.embedding.FeatureConfig],
    tf.tpu.experimental.embedding.SparseCoreEmbeddingConfig,
]:
    """Translates a Keras RS configuration to a TensorFlow TPU configuration.

    Args:
      feature_configs: The nested Keras RS feature configs.
      table_stacking: The Keras RS table stacking.
      num_replicas_in_sync: The number of replicas in sync from the strategy.

    Returns:
      A tuple containing the TensorFlow TPU feature configs and the TensorFlow
      TPU sparse core embedding config.
    """
    tables: dict[int, tf.tpu.experimental.embedding.TableConfig] = {}
    feature_configs = keras.tree.map_structure(
        lambda f: keras_to_tf_tpu_feature_config(
            f, tables, num_replicas_in_sync
        ),
        feature_configs,
    )

    # max_ids_per_chip_per_sample
    # max_ids_per_table
    # max_unique_ids_per_table

    if table_stacking is None:
        disable_table_stacking = True
    elif table_stacking == "auto":
        disable_table_stacking = False
    else:
        raise ValueError(
            f"Unsupported table stacking for Tensorflow {table_stacking}, must "
            "be 'auto' or None."
        )

    # Find alternative.
    # `initialize_tables_on_host` is set to False. Otherwise, if the
    # `TPUEmbedding` layer is built within Keras' `compute_output_spec` (meaning
    # within `call`), the tables are created within a `FuncGraph` and the
    # resulting tables are destroyed at the end of it.
    sparse_core_embedding_config = (
        tf.tpu.experimental.embedding.SparseCoreEmbeddingConfig(
            disable_table_stacking=disable_table_stacking,
            initialize_tables_on_host=False,
        )
    )

    return feature_configs, sparse_core_embedding_config


def keras_to_tf_tpu_feature_config(
    feature_config: FeatureConfig,
    tables: dict[int, tf.tpu.experimental.embedding.TableConfig],
    num_replicas_in_sync: int,
) -> tf.tpu.experimental.embedding.FeatureConfig:
    """Translates a Keras RS feature config to a TensorFlow TPU feature config.

    This creates the table config and adds it to the mapping if it doesn't exist
    in the `tables` mapping`.

    Args:
      feature_config: The Keras RS feature config to translate.
      tables: A mapping of KerasRS table config ids to TF TPU table configs.
      num_replicas_in_sync: The number of replicas in sync from the strategy.

    Returns:
      The TensorFlow TPU feature config.
    """
    if num_replicas_in_sync <= 0:
        raise ValueError(
            "`num_replicas_in_sync` must be positive, "
            f"but got {num_replicas_in_sync}."
        )

    table = tables.get(id(feature_config.table), None)
    if table is None:
        table = keras_to_tf_tpu_table_config(feature_config.table)
        tables[id(feature_config.table)] = table

    if len(feature_config.output_shape) < 2:
        raise ValueError(
            f"Invalid `output_shape` {feature_config.output_shape} in "
            f"`FeatureConfig` {feature_config}. It must have at least 2 "
            "dimensions: a batch dimension and an embedding dimension."
        )

    # Exclude last dimension, TensorFlow's TPUEmbedding doesn't want it.
    output_shape = list(feature_config.output_shape[0:-1])

    batch_size = output_shape[0]
    per_replica_batch_size: int | None = None
    if batch_size is not None:
        if batch_size % num_replicas_in_sync != 0:
            raise ValueError(
                f"Invalid `output_shape` {feature_config.output_shape} in "
                f"`FeatureConfig` {feature_config}. Batch size {batch_size} is "
                f"not a multiple of the number of TPUs {num_replicas_in_sync}."
            )
        per_replica_batch_size = batch_size // num_replicas_in_sync

    # TensorFlow's TPUEmbedding wants the per replica batch size.
    output_shape = [per_replica_batch_size] + output_shape[1:]

    # max_sequence_length
    return tf.tpu.experimental.embedding.FeatureConfig(
        name=feature_config.name,
        table=table,
        output_shape=output_shape,
    )


def keras_to_tf_tpu_table_config(
    table_config: TableConfig,
) -> tf.tpu.experimental.embedding.TableConfig:
    initializer = table_config.initializer
    if isinstance(initializer, str):
        initializer = keras.initializers.get(initializer)

    return tf.tpu.experimental.embedding.TableConfig(
        vocabulary_size=table_config.vocabulary_size,
        dim=table_config.embedding_dim,
        initializer=initializer,
        optimizer=to_tf_tpu_optimizer(table_config.optimizer),
        combiner=table_config.combiner,
        name=table_config.name,
    )


def keras_to_tf_tpu_optimizer(
    optimizer: keras.optimizers.Optimizer,
) -> TfTpuOptimizer:
    """Translates a Keras optimizer to a TensorFlow TPU `_Optimizer`.

    Args:
      optimizer: The Keras optimizer to translate.

    Returns:
      The TensorFlow TPU `_Optimizer`.
    """
    tpu_optimizer_kwargs: dict[str, Any] = {}

    # Supported keras optimizer general options.
    learning_rate = optimizer._learning_rate  # pylint: disable=protected-access
    if isinstance(
        learning_rate, keras.optimizers.schedules.LearningRateSchedule
    ):
        # Note: learning rate requires incrementing iterations in optimizer.
        tpu_optimizer_kwargs["learning_rate"] = lambda: optimizer.learning_rate
    elif callable(learning_rate):
        tpu_optimizer_kwargs["learning_rate"] = learning_rate
    else:
        learning_rate = optimizer.get_config()["learning_rate"]
        if isinstance(learning_rate, float):
            tpu_optimizer_kwargs["learning_rate"] = learning_rate
        else:
            raise ValueError(
                f"Unsupported learning rate: {learning_rate} of type"
                f" {type(learning_rate)}."
            )

    if optimizer.weight_decay is not None:
        tpu_optimizer_kwargs["weight_decay_factor"] = optimizer.weight_decay
    if optimizer.clipvalue is not None:
        tpu_optimizer_kwargs["clipvalue"] = optimizer.clipvalue
    if optimizer.gradient_accumulation_steps is not None:
        tpu_optimizer_kwargs["use_gradient_accumulation"] = True

    # Unsupported keras optimizer general options.
    if optimizer.clipnorm is not None:
        raise ValueError("Unsupported optimizer option `Optimizer.clipnorm`.")
    if optimizer.global_clipnorm is not None:
        raise ValueError(
            "Unsupported optimizer option `Optimizer.global_clipnorm`."
        )
    if optimizer.use_ema:
        raise ValueError("Unsupported optimizer option `Optimizer.use_ema`.")
    if optimizer.loss_scale_factor is not None:
        raise ValueError(
            "Unsupported optimizer option `Optimizer.loss_scale_factor`."
        )

    optimizer_mapping = None
    for optimizer_class, mapping in OPTIMIZER_MAPPINGS.items():
        # Handle subclasses of the main optimizer class.
        if isinstance(optimizer, optimizer_class):
            optimizer_mapping = mapping
            break
    if optimizer_mapping is None:
        raise ValueError(
            f"Unsupported optimizer type {type(optimizer)}. Optimizer must be "
            f"one of {list(OPTIMIZER_MAPPINGS.keys())}."
        )

    for argname in optimizer_mapping.supported_kwargs:
        tpu_optimizer_kwargs[argname] = getattr(optimizer, argname)

    for argname, disabled_value in optimizer_mapping.unsupported_kwargs.items():
        if disabled_value is None:
            if getattr(optimizer, argname) is not None:
                raise ValueError(f"Unsupported optimizer option {argname}.")
        elif getattr(optimizer, argname) != disabled_value:
            raise ValueError(f"Unsupported optimizer option {argname}.")

    return optimizer_mapping.tpu_optimizer_class(**tpu_optimizer_kwargs)


def to_tf_tpu_optimizer(
    optimizer: str | keras.optimizers.Optimizer | TfTpuOptimizer | None,
) -> TfTpuOptimizer:
    """Translates a Keras optimizer into a TensorFlow TPU `_Optimizer`.

    Args:
      optimizer: The optimizer to translate.

    Returns:
      The equivalent TensorFlow TPU `_Optimizer`.

    Raises:
      ValueError: If the optimizer or one of its argument is not supported.
    """
    if optimizer is None:
        return None
    elif isinstance(
        optimizer,
        (
            tf.tpu.experimental.embedding.SGD,
            tf.tpu.experimental.embedding.Adagrad,
            tf.tpu.experimental.embedding.Adam,
            tf.tpu.experimental.embedding.FTRL,
        ),
    ):
        return optimizer
    elif isinstance(optimizer, str):
        if optimizer == "sgd":
            return tf.tpu.experimental.embedding.SGD()
        elif optimizer == "adagrad":
            return tf.tpu.experimental.embedding.Adagrad()
        elif optimizer == "adam":
            return tf.tpu.experimental.embedding.Adam()
        elif optimizer == "ftrl":
            return tf.tpu.experimental.embedding.FTRL()
        else:
            raise ValueError(
                f"Unknown optimizer name '{optimizer}'. Please use one of "
                "'sgd', 'adagrad', 'adam', or 'ftrl'"
            )
    elif isinstance(optimizer, keras.optimizers.Optimizer):
        return keras_to_tf_tpu_optimizer(optimizer)
    else:
        raise ValueError(
            f"Unknown optimizer type {type(optimizer)}. Please pass an "
            "optimizername as a string, a subclass of keras optimizer or an "
            "instance of one of the optimizer parameter classes in "
            "`tf.tpu.experimental.embedding`."
        )


# TensorFlow to TensorFlow


def clone_tf_tpu_feature_configs(
    feature_configs: types.Nested[tf.tpu.experimental.embedding.FeatureConfig],
) -> types.Nested[tf.tpu.experimental.embedding.FeatureConfig]:
    """Clones and resolves TensorFlow TPU feature configs.

    This function clones the feature configs and resolves the table configs.

    Args:
      feature_configs: The TensorFlow TPU feature configs to clone and resolve.

    Returns:
      The cloned and resolved TensorFlow TPU feature configs.
    """
    table_configs_dict = {}

    def clone_and_resolve_tf_tpu_feature_config(
        fc: tf.tpu.experimental.embedding.FeatureConfig,
    ) -> tf.tpu.experimental.embedding.FeatureConfig:
        if fc.table not in table_configs_dict:
            table_configs_dict[fc.table] = (
                tf.tpu.experimental.embedding.TableConfig(
                    vocabulary_size=fc.table.vocabulary_size,
                    dim=fc.table.dim,
                    initializer=fc.table.initializer,
                    optimizer=to_tf_tpu_optimizer(fc.table.optimizer),
                    combiner=fc.table.combiner,
                    name=fc.table.name,
                    quantization_config=fc.table.quantization_config,
                    layout=fc.table.layout,
                )
            )
        return tf.tpu.experimental.embedding.FeatureConfig(
            table=table_configs_dict[fc.table],
            max_sequence_length=fc.max_sequence_length,
            validate_weights_and_indices=fc.validate_weights_and_indices,
            output_shape=fc.output_shape,
            name=fc.name,
        )

    return keras.tree.map_structure(
        clone_and_resolve_tf_tpu_feature_config, feature_configs
    )
