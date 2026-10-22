import keras
from keras import ops
from typing import List, Optional, Tuple

def keras_layer_norm(
    x: keras.KerasTensor,
    weight: keras.KerasTensor,
    bias: keras.KerasTensor,
    eps: float,
) -> keras.KerasTensor:
    """
    Keras 3 functional Layer Normalization implementation.
    Simulates F.layer_norm where scale/bias is applied externally.
    """
    # 1. Normalize x
    mean = ops.mean(x, axis=-1, keepdims=True)
    variance = ops.mean(ops.square(x - mean), axis=-1, keepdims=True)
    x_norm = (x - mean) / ops.sqrt(variance + eps)

    # 2. Apply weight and bias (Gamma * x_norm + Beta)
    return x_norm * weight + bias

def keras_addmm(
    bias: keras.KerasTensor,
    input: keras.KerasTensor,
    mat2: keras.KerasTensor,
) -> keras.KerasTensor:
    """Keras 3 equivalent of torch.addmm (bias + input @ mat2)."""
    return ops.add(bias, ops.matmul(input, mat2))

def hstu_compute_uqvk(
    x: keras.KerasTensor,
    norm_weight: keras.KerasTensor,
    norm_bias: keras.KerasTensor,
    norm_eps: float,
    num_heads: int,
    attn_dim: int,
    hidden_dim: int,
    uvqk_weight: keras.KerasTensor,
    uvqk_bias: keras.KerasTensor,
    kernel=None,
) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
    """
    Computes the transformed tensors U, V, Q, and K from the input X.
    """

    # 1. Normalization
    normed_x = keras_layer_norm(
        x,
        weight=norm_weight,
        bias=norm_bias,
        eps=norm_eps,
    )

    # 2. Combined Linear Projection (uvqk = bias + normed_x @ uvqk_weight)
    uvqk = keras_addmm(uvqk_bias, normed_x, uvqk_weight)

    # 3. Calculate split sizes and slice
    u_size = hidden_dim * num_heads
    v_size = hidden_dim * num_heads
    q_size = attn_dim * num_heads
    k_size = attn_dim * num_heads

    start_u = 0
    start_v = u_size
    start_q = u_size + v_size
    start_k = u_size + v_size + q_size
    L_out = ops.shape(uvqk)[0]

    u = ops.slice(uvqk, start_indices=[0, start_u], shape=[L_out, u_size])
    v = ops.slice(uvqk, start_indices=[0, start_v], shape=[L_out, v_size])
    q = ops.slice(uvqk, start_indices=[0, start_q], shape=[L_out, q_size])
    k = ops.slice(uvqk, start_indices=[0, start_k], shape=[L_out, k_size])

    # 4. Activation and Reshape
    u = ops.silu(u)
    q = ops.reshape(q, [-1, num_heads, attn_dim])
    k = ops.reshape(k, [-1, num_heads, attn_dim])
    v = ops.reshape(v, [-1, num_heads, hidden_dim])

    return u, q, k, v
