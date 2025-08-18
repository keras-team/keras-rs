from typing import Any, TypeAlias

import keras
from keras import ops

import keras_rs

Tensor: TypeAlias = Any


def _clone_initializer(
    initializer: keras.initializers.Initializer,
    seed: int | keras.random.SeedGenerator,
):
    """Clones the provided initializer with a new seed.

    This function creates a new instance of a Keras initializer from an
    existing one, but with a different seed. This is useful for ensuring
    different weights in a model are initialized with different seeds.

    Args:
        initializer: a keras.initializers.Initializer instance. The initializer
            to be cloned.
        seed: int, or a keras.random.SeedGenerator() instance. The random seed.

    Returns:
        A new `keras.initializers.Initializer` instance configured with the
        provided seed.
    """
    config = initializer.get_config()
    config.pop("seed")
    config = {**config, "seed": seed}
    initializer_class: type[keras.initializers.Initializer] = (
        initializer.__class__
    )
    return initializer_class.from_config(config)


class DLRMDCNV2(keras.Model):
    def __init__(
        self,
        large_emb_feature_configs: dict[str, keras_rs.layers.FeatureConfig],
        small_emb_features: list,
        embedding_dim: int,
        bottom_mlp_dims: list[int],
        top_mlp_dims: list[int],
        num_dcn_layers: int,
        dcn_projection_dim: int,
        seed: int | keras.random.SeedGenerator | None = None,
        dtype: str | None = None,
        name: str | None = None,
        **kwargs: Any,
    ):
        """DLRM-DCNv2 model.

        The model processes two types of input features:
        1. Dense Features: Continuous-valued features that are processed by
           a multi-layer perceptron (the "bottom MLP").
        2. Sparse Features: High-cardinality categorical features that are
           first mapped into low-dimensional embedding vectors using the
           `keras_rs.layers.DistributedEmbedding` layer. This layer is highly
           optimized for large-scale recommendation models, especially on TPUs
           with SparseCore, as it can shard large embedding tables across
           multiple accelerator chips for improved performance. On other
           hardware (GPUs, CPUs), it functions like a standard embedding layer.

        The output of the bottom MLP and the embedding vectors are then
        concatenated and fed into a DCN block for learning feature interactions.
        The output of the DCN block is then processed by another MLP
        (the "top MLP") to produce a final prediction.

        Args:
            large_emb_feature_configs: A dictionary with features names as keys
                and `keras_rs.layers.FeatureConfig` objects as values. These
                configs link features to their corresponding embedding tables
                (`keras_rs.layers.TableConfig`), specifying parameters like
                vocabulary size, embedding dimension, and hardware placement
                strategy.
            bottom_mlp_dims: A list of integers specifying the number of units
                in each layer of the bottom MLP.
            top_mlp_dims: A list of integers specifying the number of units in
                each layer of the top MLP. The last value is the final output
                dimension (e.g., 1 for binary classification).
            num_dcn_layers: The number of feature-crossing layers in the DCNv2
                block.
            dcn_projection_dim: The projection dimension used within each DCNv2
                cross-layer.
            seed: The random seed.
            dtype: Optional dtype.
            name: The name of the layer.
        """
        super().__init__(dtype=dtype, name=name, **kwargs)
        self.seed = seed

        # === Layers ====

        # Bottom MLP for encoding dense features
        self.bottom_mlp = keras.Sequential(
            self._get_mlp_layers(
                dims=bottom_mlp_dims,
                intermediate_activation="relu",
                final_activation="relu",
            ),
            name="bottom_mlp",
        )
        # Distributed embeddings for large embedding tables
        self.embedding_layer = keras_rs.layers.DistributedEmbedding(
            feature_configs=large_emb_feature_configs,
            table_stacking="auto",
            dtype=dtype,
            name="embedding_layer",
        )
        # Embedding layers for small embedding tables
        self.small_embedding_layers = None
        if small_emb_features:
            self.small_embedding_layers = [
                keras.layers.Embedding(
                    input_dim=small_emb_feature["vocabulary_size"],
                    output_dim=embedding_dim,
                    embeddings_initializer="zeros",
                    name=f"small_embedding_layer_{i}",
                )
                for i, small_emb_feature in enumerate(small_emb_features)
            ]
        # DCN for "interactions"
        self.dcn_block = DCNBlock(
            num_layers=num_dcn_layers,
            projection_dim=dcn_projection_dim,
            seed=seed,
            dtype=dtype,
            name="dcn_block",
        )
        # Top MLP for predictions
        self.top_mlp = keras.Sequential(
            self._get_mlp_layers(
                dims=top_mlp_dims,
                intermediate_activation="relu",
                final_activation="sigmoid",
            ),
            name="top_mlp",
        )

        # === Passed attributes ===
        self.large_emb_feature_configs = large_emb_feature_configs
        self.small_emb_features = small_emb_features
        self.embedding_dim = embedding_dim
        self.bottom_mlp_dims = bottom_mlp_dims
        self.top_mlp_dims = top_mlp_dims
        self.num_dcn_layers = num_dcn_layers
        self.dcn_projection_dim = dcn_projection_dim

    def call(self, inputs: dict[str, Tensor]) -> Tensor:
        """Forward pass of the model.

        Args:
            inputs: A dictionary containing `"dense_features"` and
            `"preprocessed_large_emb_features"` as keys.
        """
        # Inputs
        dense_input = inputs["dense_input"]
        large_emb_inputs = inputs["large_emb_inputs"]

        # Embed features.
        dense_output = self.bottom_mlp(dense_input)
        # jax.debug.print("dense_ouput {}", dense_output.shape)
        large_embeddings = self.embedding_layer(large_emb_inputs)
        small_embeddings = []
        if self.small_emb_features:
            small_emb_inputs = inputs["small_emb_inputs"]
            for small_emb_input, embedding_layer in zip(
                small_emb_inputs.values(), self.small_embedding_layers
            ):
                embedding = embedding_layer(small_emb_input)
                embedding = ops.sum(embedding, axis=-2)
                small_embeddings.append(embedding)

            small_embeddings = ops.concatenate(small_embeddings, axis=-1)

        # Interaction
        x = ops.concatenate(
            [dense_output, small_embeddings, *large_embeddings.values()],
            axis=-1,
        )
        # jax.debug.print("x {}", x.shape)
        x = self.dcn_block(x)

        # Predictions
        outputs = self.top_mlp(x)
        return outputs

    def _get_mlp_layers(
        self,
        dims: list[int],
        intermediate_activation: str | keras.layers.Activation,
        final_activation: str | keras.layers.Activation,
    ) -> list[keras.layers.Layer]:
        """Creates a list of Dense layers.

        Args:
            dims: list. Output dimensions of the dense layers to be created.
            intermediate_activation: string or `keras.layers.Activation`. The
                activation to be used in all layers, save the last.
            final_activation: str or `keras.layers.Activation`. The activation
                to be used in the last layer.

        Returns:
            A list of `keras.layers.Dense` layers.
        """
        initializer = keras.initializers.VarianceScaling(
            scale=1.0,
            mode="fan_in",
            distribution="uniform",
            seed=self.seed,
        )

        layers = [
            keras.layers.Dense(
                units=dim,
                activation=intermediate_activation,
                kernel_initializer=_clone_initializer(
                    initializer, seed=self.seed
                ),
                bias_initializer=_clone_initializer(
                    initializer, seed=self.seed
                ),
                dtype=self.dtype,
            )
            for dim in dims[:-1]
        ]
        layers += [
            keras.layers.Dense(
                units=dims[-1],
                activation=final_activation,
                kernel_initializer=_clone_initializer(
                    initializer, seed=self.seed
                ),
                bias_initializer=_clone_initializer(
                    initializer, seed=self.seed
                ),
                dtype=self.dtype,
            )
        ]
        return layers

    def get_config(self):
        """Returns the config of the model."""
        config = super().get_config()
        config.update(
            {
                "large_emb_feature_configs": self.large_emb_feature_configs,
                "small_emb_features": self.small_emb_features,
                "embedding_dim": self.embedding_dim,
                "bottom_mlp_dims": self.bottom_mlp_dims,
                "top_mlp_dims": self.top_mlp_dims,
                "num_dcn_layers": self.num_dcn_layers,
                "dcn_projection_dim": self.dcn_projection_dim,
                "seed": self.seed,
            }
        )
        return config


class DCNBlock(keras.layers.Layer):
    def __init__(
        self,
        num_layers: int,
        projection_dim: int,
        seed: int | keras.random.SeedGenerator,
        dtype: str | None = None,
        name: str | None = None,
        **kwargs,
    ):
        """
        A block of Deep & Cross Network V2 (DCNv2) layers.

        This layer implements the "cross network" part of the DCNv2 architecture
        by stacking multiple `keras_rs.layers.FeatureCross` layers, which learn
        feature interactions.

        Args:
            num_layers: The number of `FeatureCross` layers to stack.
            projection_dim: The dimensionality of the low-rank projection used
                within each cross layer.
            seed: The random seed for initializers.
            dtype: Optional dtype.
            name: The name of the layer.
        """
        super().__init__(dtype=dtype, name=name, **kwargs)

        # Layers
        self.layers = [
            keras_rs.layers.FeatureCross(
                projection_dim=projection_dim,
                kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
                bias_initializer="zeros",
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]

        # Passed attributes
        self.num_layers = num_layers
        self.projection_dim = projection_dim
        self.seed = seed

    def call(self, x0):
        xl = x0
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "projection_dim": self.projection_dim,
                "seed": self.seed,
            }
        )
        return config
