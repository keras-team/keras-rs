"""Configuration for TPU embedding layer."""

import dataclasses
from typing import Any

import keras

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export


@keras_rs_export("keras_rs.layers.TableConfig")
@dataclasses.dataclass(order=True)
class TableConfig:
    """Configuration for one embedding table.

    Configures one table for use by one or more `keras_rs.layers.FeatureConfig`,
    which in turn is used to configure a `keras_rs.layers.DistributedEmbedding`.

    Attributes:
        name: The name of the table. Must be defined.
        vocabulary_size: Size of the table's vocabulary (number of rows).
        embedding_dim: The embedding dimension (width) of the table.
        initializer: The initializer for the embedding weights. If not
            specified,  defaults to `truncated_normal_initializer` with mean
            `0.0` and standard deviation `1 / sqrt(embedding_dim)`.
        optimizer: The optimizer for the embedding table. Only SGD, Adagrad,
            Adam, and FTRL are supported. Note that not all of the optimizer's
            parameters are supported. Defaults to Adam.
        combiner: Specifies how to reduce if there are multiple entries in a
            single row. `mean`, `sqrtn` and `sum` are supported. `mean` is the
            default. `sqrtn` often achieves good accuracy, in particular with
            bag-of-words columns.
        placement: Where to place the embedding table. `"auto"`, which is the
            default, means that the table is placed on SparseCore if available,
            otherwise on the default device where the rest of the model is
            placed. A value of `"sparsecore"` means the table will be placed on
            the SparseCore chips and an error is raised if SparseCore is not
            available. A value of `"default_device"` means the table will be
            placed on the default device where the rest of the model is placed,
            even if SparseCore is available. The default device for the rest of
            the model is the TPU's TensorCore on TPUs, otherwise the GPU or CPU.
        max_ids_per_partition: The max number of ids per partition for the
            table. This is an input data dependent value and is required by the
            compiler to appropriately allocate memory.
        max_unique_ids_per_partition: The max number of unique ids per partition
            for the table. This is an input data dependent value and is required
            by the compiler to appropriately allocate memory.
    """

    name: str
    vocabulary_size: int
    embedding_dim: int
    initializer: str | keras.initializers.Initializer = (
        keras.initializers.VarianceScaling(mode="fan_out")
    )
    optimizer: str | keras.optimizers.Optimizer = "adam"
    combiner: str = "mean"
    placement: str = "auto"
    max_ids_per_partition: int = 256
    max_unique_ids_per_partition: int = 256

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vocabulary_size": self.vocabulary_size,
            "embedding_dim": self.embedding_dim,
            "initializer": keras.saving.serialize_keras_object(
                self.initializer
            ),
            "optimizer": keras.saving.serialize_keras_object(self.optimizer),
            "combiner": self.combiner,
            "placement": self.placement,
            "max_ids_per_partition": self.max_ids_per_partition,
            "max_unique_ids_per_partition": self.max_unique_ids_per_partition,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TableConfig":
        config = config.copy()
        config["initializer"] = keras.saving.deserialize_keras_object(
            config["initializer"]
        )
        config["optimizer"] = keras.saving.deserialize_keras_object(
            config["optimizer"]
        )
        return cls(**config)


@keras_rs_export("keras_rs.layers.FeatureConfig")
@dataclasses.dataclass(order=True)
class FeatureConfig:
    """Configuration for one embedding feature.

    Configures one feature for `keras_rs.layers.DistributedEmbedding`. Each
    feature uses a table configured via `keras_rs.layers.TableConfig` and
    multiple features can share the same table.

    Attributes:
        name: The name of the feature. Must be defined.
        table: The table in which to look up this feature.
        input_shape: The input shape of the feature. The feature fed into the
            layer has to match the shape. Note that for ragged dimensions in the
            input, the dimension provided here presents the maximum value;
            anything larger will be truncated. Also note that the first
            dimension represents the global batch size. For example, on TPU,
            this represents the total number of samples that are dispatched to
            all the TPUs connected to the current host.
        output_shape: The output shape of the feature activation. What is
            returned by the embedding layer has to match this shape.
    """

    name: str
    table: TableConfig
    input_shape: types.Shape
    output_shape: types.Shape

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "table": self.table.get_config(),
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FeatureConfig":
        config = config.copy()
        # Note: the handling of shared tables during serialization is done in
        # `DistributedEmbedding.from_config()`.
        config["table"] = TableConfig.from_config(config["table"])
        return cls(**config)
