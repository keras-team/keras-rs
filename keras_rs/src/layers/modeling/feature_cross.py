from typing import Any, Optional, Text, Union

import keras
from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.utils.keras_utils import clone_initializer


@keras_rs_export("keras_rs.layers.FeatureCross")
class FeatureCross(keras.layers.Layer):
    """FeatureCross layer in Deep & Cross Network (DCN).

    A layer that creates explicit and bounded-degree feature interactions
    efficiently. The `call` method accepts `inputs` as a tuple of size 2
    tensors. The first input `x0` is the base layer that contains the original
    features (usually the embedding layer); the second input `xi` is the output
    of the previous `FeatureCross` layer in the stack, i.e., the i-th
    `FeatureCross` layer. For the first `FeatureCross` layer in the stack,
    `x0 = xi`.

    The output is `x_{i+1} = x0 .* (W * x_i + bias + diag_scale * x_i) + x_i`,
    where .* designates elementwise multiplication, W could be a full-rank
    matrix, or a low-rank matrix `U*V` to reduce the computational cost, and
    diag_scale increases the diagonal of W to improve training stability (
    especially for the low-rank case).

    References:
        - [R. Wang et al.](https://arxiv.org/abs/2008.13535)
        - [R. Wang et al.](https://arxiv.org/abs/1708.05123)

    Example:

        ```python
        # after embedding layer in a functional model:
        input = keras.Input(shape=(None,), name='index', dtype="int64")
        x0 = keras.layers.Embedding(input_dim=32, output_dim=6)
        x1 = FeatureCross()(x0, x0)
        x2 = FeatureCross()(x0, x1)
        logits = keras.layers.Dense(units=10)(x2)
        model = keras.Model(input, logits)
        ```

    Args:
        projection_dim: project dimension to reduce the computational cost.
          Default is `None` such that a full (`input_dim` by `input_dim`) matrix
          W is used. If enabled, a low-rank matrix W = U*V will be used, where U
          is of size `input_dim` by `projection_dim` and V is of size
          `projection_dim` by `input_dim`. `projection_dim` need to be smaller
          than `input_dim`/2 to improve the model efficiency. In practice, we've
          observed that `projection_dim` = d/4 consistently preserved the
          accuracy of a full-rank version.
        diag_scale: a non-negative float used to increase the diagonal of the
          kernel W by `diag_scale`, that is, W + diag_scale * I, where I is an
          identity matrix.
        use_bias: whether to add a bias term for this layer. If set to False,
          no bias term will be used.
        pre_activation: Activation applied to output matrix of the layer, before
          multiplication with the input. Can be used to control the scale of the
          layer's outputs and improve stability.
        kernel_initializer: Initializer to use on the kernel matrix.
        bias_initializer: Initializer to use on the bias vector.
        kernel_regularizer: Regularizer to use on the kernel matrix.
        bias_regularizer: Regularizer to use on bias vector.

    Input shape: A tuple of 2 (batch_size, `input_dim`) dimensional inputs.
    Output shape: A single (batch_size, `input_dim`) dimensional output.
    """

    def __init__(
        self,
        projection_dim: Optional[int] = None,
        diag_scale: Optional[float] = 0.0,
        use_bias: bool = True,
        pre_activation: Optional[Union[str, keras.layers.Activation]] = None,
        kernel_initializer: Union[
            Text, keras.initializers.Initializer
        ] = "glorot_uniform",
        bias_initializer: Union[Text, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Union[
            Text, None, keras.regularizers.Regularizer
        ] = None,
        bias_regularizer: Union[
            Text, None, keras.regularizers.Regularizer
        ] = None,
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

    def build(self, input_shape: types.TensorShape) -> None:
        last_dim = input_shape[-1]

        dense_layer_args = {
            "units": last_dim,
            "activation": self.pre_activation,
            "use_bias": self.use_bias,
            "kernel_initializer": clone_initializer(self.kernel_initializer),
            "bias_initializer": clone_initializer(self.bias_initializer),
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
        }

        if self.projection_dim is not None:
            self.down_proj_dense = keras.layers.Dense(
                units=self.projection_dim,
                use_bias=False,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                kernel_regularizer=self.kernel_regularizer,
                dtype=self.dtype_policy,
            )

        self.dense = keras.layers.Dense(
            **dense_layer_args,
            dtype=self.dtype_policy,
        )

        self.built = True

    def call(
        self, x0: types.Tensor, x: Optional[types.Tensor] = None
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

        if not self.built:
            self.build(x0.shape)

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
            output = output + self.diag_scale * x

        return x0 * output + x

    def get_config(self) -> Any:
        config = super().get_config()

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

        # Typecast config to `dict`. This is not really needed,
        # but `typing` throws an error if we don't do this.
        config = dict(config)
        return config
