from typing import Any, Callable, Sequence, TypeAlias

import keras
import tensorflow as tf

from keras_rs.src import types
from keras_rs.src.layers.embedding import base_distributed_embedding
from keras_rs.src.layers.embedding import distributed_embedding_config
from keras_rs.src.layers.embedding.tensorflow import config_conversion
from keras_rs.src.utils import keras_utils

FeatureConfig = distributed_embedding_config.FeatureConfig
TableConfig = distributed_embedding_config.TableConfig

# Placeholder of tf.tpu.experimental.embedding._Optimizer which is not exposed.
TfTpuOptimizer: TypeAlias = Any


GRADIENT_TRAP_DUMMY_NAME = "_gradient_trap_dummy"

EMBEDDING_FEATURE_V1 = tf.tpu.experimental.HardwareFeature.EmbeddingFeature.V1
EMBEDDING_FEATURE_V2 = tf.tpu.experimental.HardwareFeature.EmbeddingFeature.V2
UNSUPPORTED = tf.tpu.experimental.HardwareFeature.EmbeddingFeature.UNSUPPORTED


class DistributedEmbedding(base_distributed_embedding.DistributedEmbedding):
    """TensorFlow implementation of the TPU embedding layer."""

    def __init__(
        self,
        feature_configs: types.Nested[
            FeatureConfig | tf.tpu.experimental.embedding.FeatureConfig
        ],
        *,
        table_stacking: (
            str | Sequence[str] | Sequence[Sequence[str]]
        ) = "auto",
        **kwargs: Any,
    ) -> None:
        # Intercept arguments that are supported only on TensorFlow.
        self._optimizer = kwargs.pop("optimizer", None)
        self._pipeline_execution_with_tensor_core = kwargs.pop(
            "pipeline_execution_with_tensor_core", False
        )
        self._sparse_core_embedding_config = kwargs.pop(
            "sparse_core_embedding_config", None
        )

        # Mark as True by default for `_verify_input_shapes`. This will be
        # updated in `_sparsecore_init` if applicable.
        self._using_keras_rs_configuration = True

        super().__init__(
            feature_configs, table_stacking=table_stacking, **kwargs
        )

    def _is_tpu_strategy(self, strategy: tf.distribute.Strategy) -> bool:
        return isinstance(
            strategy,
            (tf.distribute.TPUStrategy, tf.distribute.experimental.TPUStrategy),
        )

    def _has_sparsecore(self) -> bool:
        strategy = tf.distribute.get_strategy()
        if self._is_tpu_strategy(strategy):
            tpu_embedding_feature = (
                strategy.extended.tpu_hardware_feature.embedding_feature
            )
            return tpu_embedding_feature in (
                EMBEDDING_FEATURE_V2,
                EMBEDDING_FEATURE_V1,
            )
        return False

    @keras_utils.no_automatic_dependency_tracking
    def _sparsecore_init(
        self,
        feature_configs: dict[
            str,
            FeatureConfig | tf.tpu.experimental.embedding.FeatureConfig,
        ],
        table_stacking: str | Sequence[str] | Sequence[Sequence[str]],
    ) -> None:
        self._table_stacking = table_stacking

        strategy = tf.distribute.get_strategy()
        if not self._is_tpu_strategy(strategy):
            raise ValueError(
                "Placement to sparsecore was requested, however, we are not "
                "running under a TPU strategy."
            )

        tpu_embedding_feature = (
            strategy.extended.tpu_hardware_feature.embedding_feature
        )

        self._using_keras_rs_configuration = isinstance(
            next(iter(feature_configs.values())), FeatureConfig
        )
        if self._using_keras_rs_configuration:
            if self._sparse_core_embedding_config is not None:
                raise ValueError(
                    "The `sparse_core_embedding_config` argument is only "
                    "supported when using "
                    "`tf.tpu.experimental.embedding.FeatureConfig` instances "
                    "for the configuration."
                )
            self._tpu_feature_configs, self._sparse_core_embedding_config = (
                config_conversion.keras_to_tf_tpu_configuration(
                    feature_configs,
                    table_stacking,
                    strategy.num_replicas_in_sync,
                )
            )
            if tpu_embedding_feature == EMBEDDING_FEATURE_V1:
                # Remove auto-generated SparseCoreEmbeddingConfig, which is not
                # used.
                self._sparse_core_embedding_config = None
        else:
            if table_stacking != "auto":
                raise ValueError(
                    "The `table_stacking` argument is not supported when using "
                    "`tf.tpu.experimental.embedding.FeatureConfig` for the "
                    "configuration. You can use the `disable_table_stacking` "
                    "attribute of "
                    "`tf.tpu.experimental.embedding.SparseCoreEmbeddingConfig` "
                    "to disable table stacking."
                )
            if (
                tpu_embedding_feature == EMBEDDING_FEATURE_V1
                and self._sparse_core_embedding_config is not None
            ):
                raise ValueError(
                    "The `sparse_core_embedding_config` argument is not "
                    "supported with this TPU generation."
                )
            self._tpu_feature_configs = (
                config_conversion.clone_tf_tpu_feature_configs(feature_configs)
            )

        self._tpu_optimizer = config_conversion.to_tf_tpu_optimizer(
            self._optimizer
        )

        if tpu_embedding_feature == EMBEDDING_FEATURE_V1:
            self._tpu_embedding = tf.tpu.experimental.embedding.TPUEmbedding(
                self._tpu_feature_configs,
                self._tpu_optimizer,
                self._pipeline_execution_with_tensor_core,
            )
            self._v1_call_id = 0
        elif tpu_embedding_feature == EMBEDDING_FEATURE_V2:
            self._tpu_embedding = tf.tpu.experimental.embedding.TPUEmbeddingV2(
                self._tpu_feature_configs,
                self._tpu_optimizer,
                self._pipeline_execution_with_tensor_core,
                self._sparse_core_embedding_config,
            )
        elif tpu_embedding_feature == UNSUPPORTED:
            raise ValueError(
                "Placement to sparsecore was requested, however, this TPU does "
                "not support it."
            )
        elif tpu_embedding_feature != UNSUPPORTED:
            raise ValueError(
                f"Unsupported TPU embedding feature: {tpu_embedding_feature}."
            )

        # We need at least one trainable variable for the gradient trap to work.
        # Note that the Python attribute name "_gradient_trap_dummy" should
        # match the name of the variable GRADIENT_TRAP_DUMMY_NAME.
        self._gradient_trap_dummy = self.add_weight(
            name=GRADIENT_TRAP_DUMMY_NAME,
            shape=(1,),
            initializer=tf.zeros_initializer(),
            trainable=True,
            dtype=tf.float32,
        )

    def compute_output_shape(
        self, input_shapes: types.Nested[types.Shape]
    ) -> types.Nested[types.Shape]:
        if self._using_keras_rs_configuration:
            return super().compute_output_shape(input_shapes)

        def _compute_output_shape(
            feature_config: tf.tpu.experimental.embedding.FeatureConfig,
            input_shape: types.Shape,
        ) -> types.Shape:
            if len(input_shape) < 1:
                raise ValueError(
                    f"Received input shape {input_shape}. Rank must be 1 or "
                    "above."
                )
            max_sequence_length: int = feature_config.max_sequence_length
            embed_dim = feature_config.table.dim
            if (
                feature_config.output_shape is not None
                and feature_config.output_shape.rank is not None
            ):
                return tuple(feature_config.output_shape.as_list())
            elif (
                len(input_shape) == 2
                and input_shape[-1] != 1
                and max_sequence_length > 0
            ):
                # Update the input shape with the max sequence length. Only
                # update when:
                # 1. Input feature is 2D ragged or sparse tensor.
                # 2. Output shape is not set and max sequence length is set.
                return tuple(input_shape[:-1]) + (
                    max_sequence_length,
                    embed_dim,
                )
            elif len(input_shape) == 1:
                return tuple(input_shape) + (embed_dim,)
            else:
                return tuple(input_shape[:-1]) + (embed_dim,)

        output_shapes: types.Nested[types.Shape] = (
            keras.tree.map_structure_up_to(
                self._feature_configs,
                _compute_output_shape,
                self._feature_configs,
                input_shapes,
            )
        )
        return output_shapes

    def _sparsecore_build(self, input_shapes: dict[str, types.Shape]) -> None:
        if isinstance(
            self._tpu_embedding, tf.tpu.experimental.embedding.TPUEmbedding
        ):
            tf_input_shapes = keras.tree.map_shape_structure(
                tf.TensorShape, input_shapes
            )
            tpu_embedding_build = tf.autograph.to_graph(
                self._tpu_embedding.build, recursive=False
            )
            tpu_embedding_build(
                self._tpu_embedding, per_replica_input_shapes=tf_input_shapes
            )
        elif isinstance(
            self._tpu_embedding, tf.tpu.experimental.embedding.TPUEmbeddingV2
        ):
            self._tpu_embedding.build()

    def _sparsecore_call(
        self,
        inputs: dict[str, types.Tensor],
        weights: dict[str, types.Tensor] | None = None,
        training: bool = False,
    ) -> dict[str, types.Tensor]:
        del training  # Unused.
        strategy = tf.distribute.get_strategy()
        if not self._is_tpu_strategy(strategy):
            raise RuntimeError(
                "DistributedEmbedding needs to be called under a TPUStrategy "
                "for features placed on the embedding feature but is being "
                f"called under strategy {strategy}. Please use `strategy.run` "
                "when calling this layer."
            )
        if isinstance(
            self._tpu_embedding, tf.tpu.experimental.embedding.TPUEmbedding
        ):
            return self._tpu_embedding_lookup_v1(
                self._tpu_embedding, inputs, weights
            )
        elif isinstance(
            self._tpu_embedding, tf.tpu.experimental.embedding.TPUEmbeddingV2
        ):
            return self._tpu_embedding_lookup_v2(
                self._tpu_embedding, inputs, weights
            )
        else:
            raise ValueError(
                "DistributedEmbedding is receiving features to lookup on the "
                "TPU embedding feature but no such feature was configured."
            )

    def _sparsecore_get_embedding_tables(self) -> dict[str, types.Tensor]:
        tables: dict[str, types.Tensor] = {}
        strategy = tf.distribute.get_strategy()
        if not self._is_tpu_strategy(strategy):
            raise RuntimeError(
                "`DistributedEmbedding.get_embedding_tables` needs to be "
                "called under the TPUStrategy that DistributedEmbedding was "
                f"created with, but is being called under strategy {strategy}. "
                "Please use `with strategy.scope()` when calling "
                "`get_embedding_tables`."
            )

        tpu_hardware = strategy.extended.tpu_hardware_feature
        num_sc_per_device = tpu_hardware.num_embedding_devices_per_chip
        num_shards = strategy.num_replicas_in_sync * num_sc_per_device

        def populate_table(
            feature_config: tf.tpu.experimental.embedding.FeatureConfig,
        ) -> None:
            table_name = feature_config.table.name
            if table_name in tables:
                return

            embedding_dim = feature_config.table.dim
            table = self._tpu_embedding.embedding_tables[table_name]

            # This table has num_sparse_cores mod shards, so we need to slice,
            # reconcat and reshape.
            table_shards = [
                shard.numpy()[:, :embedding_dim] for shard in table.values
            ]
            full_table = keras.ops.concatenate(table_shards, axis=0)
            full_table = keras.ops.concatenate(
                keras.ops.split(full_table, num_shards, axis=0), axis=1
            )
            full_table = keras.ops.reshape(full_table, [-1, embedding_dim])
            tables[table_name] = full_table[
                : feature_config.table.vocabulary_size, :
            ]

        keras.tree.map_structure(populate_table, self._tpu_feature_configs)
        return tables

    def _verify_input_shapes(
        self, input_shapes: types.Nested[types.Shape]
    ) -> None:
        if self._using_keras_rs_configuration:
            return super()._verify_input_shapes(input_shapes)
        # `tf.tpu.experimental.embedding.FeatureConfig` does not provide any
        # information about the input shape, so there is nothing to verify.

    def _tpu_embedding_lookup_v1(
        self,
        tpu_embedding: tf.tpu.experimental.embedding.TPUEmbedding,
        inputs: dict[str, types.Tensor],
        weights: dict[str, types.Tensor] | None = None,
    ) -> dict[str, types.Tensor]:
        # Each call to this function increments the _v1_call_id by 1, this
        # allows us to tag each of the main embedding ops with this call id so
        # that we know during graph rewriting passes which ops correspond to the
        # same layer call.
        self._v1_call_id += 1
        name = str(self._v1_call_id)

        # Set training to true, even during eval. When name is set, this will
        # trigger a pass that updates the training based on if there is a send
        # gradients with the same name.
        tpu_embedding.enqueue(inputs, weights, training=True, name=name)

        @tf.custom_gradient  # type: ignore
        def gradient_trap(
            dummy: types.Tensor,
        ) -> tuple[
            list[types.Tensor], Callable[[tuple[types.Tensor]], types.Tensor]
        ]:
            """Register a gradient function for activation."""
            activations = tpu_embedding.dequeue(name=name)

            def grad(*grad_wrt_activations: types.Tensor) -> types.Tensor:
                """Gradient function."""
                # Since the output were flattened, the gradients are also
                # flattened. Pack them back into the correct nested structure.
                gradients = tf.nest.pack_sequence_as(
                    self._placement_to_path_to_feature_config["sparsecore"],
                    grad_wrt_activations,
                )
                tpu_embedding.apply_gradients(gradients, name=name)

                # This is the gradient for the input variable.
                return tf.zeros_like(dummy)

            # Custom gradient functions don't like nested structures of tensors,
            # so we flatten them here.
            return tf.nest.flatten(activations), grad

        activations_with_trap = gradient_trap(self._gradient_trap_dummy.value)
        result: dict[str, types.Tensor] = tf.nest.pack_sequence_as(
            self._placement_to_path_to_feature_config["sparsecore"],
            activations_with_trap,
        )
        return result

    def _tpu_embedding_lookup_v2(
        self,
        tpu_embedding: tf.tpu.experimental.embedding.TPUEmbeddingV2,
        inputs: dict[str, types.Tensor],
        weights: dict[str, types.Tensor] | None = None,
    ) -> dict[str, types.Tensor]:
        @tf.custom_gradient  # type: ignore
        def gradient_trap(
            dummy: types.Tensor,
        ) -> tuple[
            list[types.Tensor], Callable[[tuple[types.Tensor]], types.Tensor]
        ]:
            """Register a gradient function for activation."""
            activations, preserved_result = tpu_embedding(inputs, weights)

            def grad(*grad_wrt_activations: types.Tensor) -> types.Tensor:
                """Gradient function."""
                # Since the output were flattened, the gradients are also
                # flattened. Pack them back into the correct nested structure.
                gradients = tf.nest.pack_sequence_as(
                    self._placement_to_path_to_feature_config["sparsecore"],
                    grad_wrt_activations,
                )
                tpu_embedding.apply_gradients(
                    gradients, preserved_outputs=preserved_result
                )
                # This is the gradient for the input variable.
                return tf.zeros_like(dummy)

            # Custom gradient functions don't like nested structures of tensors,
            # so we flatten them here.
            return tf.nest.flatten(activations), grad

        activations_with_trap = gradient_trap(self._gradient_trap_dummy)
        result: dict[str, types.Tensor] = tf.nest.pack_sequence_as(
            self._placement_to_path_to_feature_config["sparsecore"],
            activations_with_trap,
        )
        return result

    def _trackable_children(
        self, save_type: str = "checkpoint", **kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        # Remove dummy variable, we don't want it in checkpoints.
        children: dict[str, Any] = super()._trackable_children(
            save_type, **kwargs
        )
        children.pop(GRADIENT_TRAP_DUMMY_NAME, None)
        return children


DistributedEmbedding.__doc__ = (
    base_distributed_embedding.DistributedEmbedding.__doc__
)
