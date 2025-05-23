from typing import Any

import keras
from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.utils.keras_utils import clone_initializer


@keras_rs_export("keras_rs.layers.FeatureCross")
class FeatureCross(keras.layers.Layer):
    """FeatureCross layer in Deep & Cross Network (DCN).

    A layer that creates explicit and bounded-degree feature interactions
    efficiently. The `call` method accepts two inputs: `x0` contains the
    original features; the second input `xi` is the output of the previous
    `FeatureCross` layer in the stack, i.e., the i-th `FeatureCross` layer.
    For the first `FeatureCross` layer in the stack, `x0 = xi`.

    The output is `x_{i+1} = x0 .* (W * x_i + bias + diag_scale * x_i) + x_i`,
    where .* denotes element-wise multiplication. W could be a full-rank
    matrix, or a low-rank matrix `U*V` to reduce the computational cost, and
    `diag_scale` increases the diagonal of W to improve training stability (
    especially for the low-rank case).

    Args:
        projection_dim: int. Dimension for down-projecting the input to reduce
            computational cost. If `None` (default), the full matrix, `W`
            (with shape `(input_dim, input_dim)`) is used. Otherwise, a low-rank
            matrix `W = U*V` will be used, where `U` is of shape
            `(input_dim, projection_dim)` and `V` is of shape
            `(projection_dim, input_dim)`. `projection_dim` need to be smaller
            than `input_dim//2` to improve the model efficiency. In practice,
            we've observed that `projection_dim = input_dim//4` consistently
            preserved the accuracy of a full-rank version.
        diag_scale: non-negative float. Used to increase the diagonal of the
            kernel W by `diag_scale`, i.e., `W + diag_scale * I`, where I is the
            identity matrix. Defaults to `None`.
        use_bias: bool. Whether to add a bias term for this layer. Defaults to
            `True`.
        pre_activation: string or `keras.activations`. Activation applied to
            output matrix of the layer, before multiplication with the input.
            Can be used to control the scale of the layer's outputs and
            improve stability. Defaults to `None`.
        kernel_initializer: string or `keras.initializers` initializer.
            Initializer to use for the kernel matrix. Defaults to
            `"glorot_uniform"`.
        bias_initializer: string or `keras.initializers` initializer.
            Initializer to use for the bias vector. Defaults to `"ones"`.
        kernel_regularizer: string or `keras.regularizer` regularizer.
            Regularizer to use for the kernel matrix.
        bias_regularizer: string or `keras.regularizer` regularizer.
            Regularizer to use for the bias vector.
        **kwargs: Args to pass to the base class.

    Example:

    ```python
    # 1. Simple forward pass
    batch_size = 2
    embedding_dim = 32
    feature1 = np.random.randn(batch_size, embedding_dim)
    feature2 = np.random.randn(batch_size, embedding_dim)
    crossed_features = keras_rs.layers.FeatureCross()(feature1, feature2)

    # 2. After embedding layer in a model
    vocabulary_size = 32
    embedding_dim = 6

    # Create a simple model containing the layer.
    inputs = keras.Input(shape=(), name='indices', dtype="int32")
    x0 = keras.layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dim
    )(inputs)
    x1 = keras_rs.layers.FeatureCross()(x0, x0)
    x2 = keras_rs.layers.FeatureCross()(x0, x1)
    logits = keras.layers.Dense(units=10)(x2)
    model = keras.Model(inputs, logits)

    # Call the model on the inputs.
    batch_size = 2
    input_data = np.random.randint(0, vocabulary_size, size=(batch_size,))
    outputs = model(input_data)
    ```

    References:
    - [R. Wang et al.](https://arxiv.org/abs/2008.13535)
    - [R. Wang et al.](https://arxiv.org/abs/1708.05123)
    """

    def __init__(
        self,
        projection_dim: int | None = None,
        diag_scale: float | None = 0.0,
        use_bias: bool = True,
        pre_activation: str | keras.layers.Activation | None = None,
        kernel_initializer: (
            str | keras.initializers.Initializer
        ) = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "zeros",
        kernel_regularizer: (
            str | None | keras.regularizers.Regularizer
        ) = None,
        bias_regularizer: (str | None | keras.regularizers.Regularizer) = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Passed args.
        self.projection_dim = projection_dim
        self.diag_scale = diag_scale
        self.use_bias = use_bias
        self.pre_activation = keras.activations.get(pre_activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Other args.
        self.supports_masking = True

        if self.diag_scale is not None and self.diag_scale < 0.0:
            raise ValueError(
                "`diag_scale` should be non-negative. Received: "
                f"`diag_scale={self.diag_scale}`"
            )

    def build(self, input_shape: types.Shape) -> None:
        last_dim = input_shape[-1]

        if self.projection_dim is not None:
            self.down_proj_dense = keras.layers.Dense(
                units=self.projection_dim,
                use_bias=False,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                kernel_regularizer=self.kernel_regularizer,
                dtype=self.dtype_policy,
            )

        self.dense = keras.layers.Dense(
            units=last_dim,
            activation=self.pre_activation,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            dtype=self.dtype_policy,
        )

        self.built = True

    def call(
        self, x0: types.Tensor, x: types.Tensor | None = None
    ) -> types.Tensor:
        """Forward pass of the cross layer.

        Args:
            x0: a Tensor. The input to the cross layer. N-rank tensor
                with shape `(batch_size, ..., input_dim)`.
            x: a Tensor. Optional. If provided, the layer will compute
                crosses between x0 and x. Otherwise, the layer will
                compute crosses between x0 and itself. Should have the same
                shape as `x0`.

        Returns:
            Tensor of crosses, with the same shape as `x0`.
        """

        if x is None:
            x = x0

        if x0.shape != x.shape:
            raise ValueError(
                "`x0` and `x` should have the same shape. Received: "
                f"`x.shape` = {x.shape}, `x0.shape` = {x0.shape}"
            )

        # Project to a lower dimension.
        if self.projection_dim is None:
            output = x
        else:
            output = self.down_proj_dense(x)

        output = self.dense(output)

        output = ops.cast(output, self.compute_dtype)

        if self.diag_scale:
            output = ops.add(output, ops.multiply(self.diag_scale, x))

        return ops.add(ops.multiply(x0, output), x)

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = super().get_config()

        config.update(
            {
                "projection_dim": self.projection_dim,
                "diag_scale": self.diag_scale,
                "use_bias": self.use_bias,
                "pre_activation": keras.activations.serialize(
                    self.pre_activation
                ),
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": keras.regularizers.serialize(
                    self.bias_regularizer
                ),
            }
        )

        return config
