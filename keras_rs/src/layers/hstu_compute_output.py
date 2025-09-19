import keras
from keras import ops
from typing import List, Optional, Tuple

def keras_norm_mul_dropout(
    x: keras.KerasTensor,
    u: keras.KerasTensor,
    weight: keras.KerasTensor,
    bias: keras.KerasTensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool = False,
    concat_ux: bool = False,
    group_norm: bool = False,
    num_heads: int = 1,
    linear_dim: int = -1,
) -> keras.KerasTensor:
    """
    Keras 3 equivalent of pytorch_norm_mul_dropout.
    Applies normalization, element-wise multiplication with u, and dropout.
    Assumes keras_layer_norm is available (though the logic is inlined here).
    """
    x = ops.convert_to_tensor(x, dtype='float32')
    u = ops.convert_to_tensor(u, dtype='float32')

    if silu_u:
        u = ops.silu(u)

    if group_norm:
        raise NotImplementedError("Group Norm path not suitable for simple Keras ops conversion.")
    else:
        # Functional Layer Normalization (Simulated keras_layer_norm)
        mean = ops.mean(x, axis=-1, keepdims=True)
        variance = ops.mean(ops.square(x - mean), axis=-1, keepdims=True)
        x_norm = (x - mean) / ops.sqrt(variance + eps)

        # Apply weight and bias (Gamma * x_norm + Beta)
        y_norm = x_norm * weight + bias

        # Apply u multiplication (Element-wise gating)
        y = u * y_norm

    if concat_ux:
        y = ops.concatenate([u, x, y], axis=1)

    # Dropout (using Keras layer for correct training=True/False behavior)
    y = keras.layers.Dropout(dropout_ratio)(y, training=training)

    return ops.cast(y, dtype=x.dtype)

def keras_hstu_compute_output(
    attn: keras.KerasTensor,
    u: keras.KerasTensor,
    x: keras.KerasTensor,
    norm_weight: keras.KerasTensor,
    norm_bias: keras.KerasTensor,
    output_weight: keras.KerasTensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool = False,
    concat_ux: bool = False,
    group_norm: bool = False,
    num_heads: int = 1,
    linear_dim: int = -1,
) -> keras.KerasTensor:
    """
    Core kernel for the final residual block calculation (Attn Output -> Norm/Dropout -> MatMul -> Residual Add).
    """
    y = keras_norm_mul_dropout(
        x=attn,
        u=u,
        weight=norm_weight,
        bias=norm_bias,
        eps=eps,
        dropout_ratio=dropout_ratio,
        training=training,
        silu_u=silu_u,
        concat_ux=concat_ux,
        group_norm=group_norm,
        num_heads=num_heads,
        linear_dim=linear_dim,
    )

    # Final output: Residual addition of input (x) and transformed attention output (y @ output_weight)
    output = ops.add(x, ops.matmul(y, output_weight))

    return output

def hstu_compute_output(
    attn: keras.KerasTensor,
    u: keras.KerasTensor,
    x: keras.KerasTensor,
    norm_weight: keras.KerasTensor,
    norm_bias: keras.KerasTensor,
    norm_eps: float,
    output_weight: keras.KerasTensor,
    num_heads: int,
    linear_dim: int,
    dropout_ratio: float,
    training: bool,
    concat_ux: bool,
    group_norm: bool,
    recompute_y_in_backward: bool,
) -> keras.KerasTensor:
    """
    Top-level wrapper for the output computation, delegates to the core Keras kernel.
    """
    return keras_hstu_compute_output(
        attn=attn,
        u=u,
        x=x,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        output_weight=output_weight,
        eps=norm_eps,
        dropout_ratio=dropout_ratio,
        training=training,
        silu_u=False,
        concat_ux=concat_ux,
        group_norm=group_norm,
        num_heads=num_heads,
        linear_dim=linear_dim,
    )
