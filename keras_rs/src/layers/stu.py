import abc
from typing import List, Optional, Tuple
import keras
from keras import ops
from keras import layers

from keras_rs.src.layers.common import fx_unwrap_optional_tensor
from keras_rs.src.layers.hstu_compute_output import hstu_compute_uqvk, hstu_compute_output
from keras_rs.src.layers.hstu_preprocess_attention import keras_hstu_preprocess_and_attention
from keras_rs.src.layers.hstu_mha_attention import delta_hstu_mha
from keras_rs.src.layers.jagged_tensors import split_2D_jagged, concat_2D_jagged


class STULayerConfig:
    def __init__(self, embedding_dim: int, num_heads: int, hidden_dim: int, attention_dim: int,
                 output_dropout_ratio: float = 0.3, causal: bool = True, target_aware: bool = True, 
                 max_attn_len: Optional[int] = None, attn_alpha: Optional[float] = None, 
                 use_group_norm: bool = False, recompute_normed_x: bool = True, 
                 recompute_uvqk: bool = True, recompute_y: bool = True, 
                 sort_by_length: bool = True, contextual_seq_len: int = 0):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.output_dropout_ratio = output_dropout_ratio
        self.causal = causal
        self.target_aware = target_aware
        self.max_attn_len = max_attn_len
        self.attn_alpha = attn_alpha
        self.use_group_norm = use_group_norm
        self.recompute_normed_x = recompute_normed_x
        self.recompute_uvqk = recompute_uvqk
        self.recompute_y = recompute_y
        self.sort_by_length = sort_by_length
        self.contextual_seq_len = contextual_seq_len


def _update_kv_cache(
    max_seq_len: int, seq_offsets: keras.KerasTensor, k: Optional[keras.KerasTensor], v: Optional[keras.KerasTensor], max_kv_caching_len: int, kv_caching_lengths: Optional[keras.KerasTensor], orig_k_cache: Optional[keras.KerasTensor], orig_v_cache: Optional[keras.KerasTensor], orig_max_kv_caching_len: int, orig_kv_caching_offsets: Optional[keras.KerasTensor],
) -> Tuple[Optional[keras.KerasTensor], Optional[keras.KerasTensor], int, Optional[keras.KerasTensor]]:
    
    if kv_caching_lengths is not None:
        # Keras equivalent of asynchronous_complete_cumsum
        kv_caching_offsets = ops.cast(ops.cumsum(kv_caching_lengths, exclusive=True), dtype="int32")
        delta_offsets = seq_offsets - kv_caching_offsets
        
        # NOTE: split_2D_jagged is available from jagged_tensors.py
        k_cache, _ = split_2D_jagged(max_seq_len=max_seq_len, values=ops.reshape(fx_unwrap_optional_tensor(k), [-1, ops.shape(k)[-1]]), max_len_left=None, max_len_right=None, offsets_left=kv_caching_offsets, offsets_right=delta_offsets)
        v_cache, _ = split_2D_jagged(max_seq_len=max_seq_len, values=ops.reshape(fx_unwrap_optional_tensor(v), [-1, ops.shape(v)[-1]]), max_len_left=None, max_len_right=None, offsets_left=kv_caching_offsets, offsets_right=delta_offsets)

        if max_kv_caching_len == 0: 
            max_kv_caching_len = ops.convert_to_numpy(ops.cast(ops.max(kv_caching_lengths), dtype="int32")).item()
        return (k_cache, v_cache, max_kv_caching_len, kv_caching_offsets)
    else:
        return (orig_k_cache, orig_v_cache, orig_max_kv_caching_len, orig_kv_caching_offsets)


def _construct_full_kv(
    delta_k: keras.KerasTensor, delta_v: keras.KerasTensor, k_cache: keras.KerasTensor, v_cache: keras.KerasTensor, max_kv_caching_len: int, kv_caching_offsets: keras.KerasTensor,
) -> Tuple[keras.KerasTensor, keras.KerasTensor, int, keras.KerasTensor]:
    L = ops.shape(delta_k)[0]
    B = ops.shape(kv_caching_offsets)[0] - 1
    delta_size = L // B

    # NOTE: concat_2D_jagged is available from jagged_tensors.py
    full_k = concat_2D_jagged(max_seq_len=max_kv_caching_len + delta_size, values_left=k_cache, values_right=delta_k, max_len_left=max_kv_caching_len, max_len_right=delta_size, offsets_left=kv_caching_offsets, offsets_right=None)
    full_v = concat_2D_jagged(max_seq_len=max_kv_caching_len + delta_size, values_left=v_cache, values_right=delta_v, max_len_left=max_kv_caching_len, max_len_right=delta_size, offsets_left=kv_caching_offsets, offsets_right=None)
    
    # Calculate new combined offsets
    delta_size_broadcast = delta_size * ops.arange(B + 1, dtype=kv_caching_offsets.dtype)
    full_kv_caching_offsets = kv_caching_offsets + delta_size_broadcast
    
    return (full_k, full_v, max_kv_caching_len + delta_size, full_kv_caching_offsets)


class STU(layers.Layer, abc.ABC):
    """Abstract base class for STU layers."""
    @abc.abstractmethod
    def cached_forward(self, delta_x: keras.KerasTensor, num_targets: keras.KerasTensor, max_kv_caching_len: int = 0, kv_caching_lengths: Optional[keras.KerasTensor] = None, training: Optional[bool] = None,) -> keras.KerasTensor: pass
    @abc.abstractmethod
    def call(self, x: keras.KerasTensor, x_lengths: keras.KerasTensor, x_offsets: keras.KerasTensor, max_seq_len: int, num_targets: keras.KerasTensor, max_kv_caching_len: int = 0, kv_caching_lengths: Optional[keras.KerasTensor] = None, training: Optional[bool] = None,) -> keras.KerasTensor: pass


class STULayer(layers.Layer):
    # Initialize cache properties on the instance
    max_kv_caching_len: int = 0
    k_cache: Optional[keras.KerasTensor] = None
    v_cache: Optional[keras.KerasTensor] = None
    kv_caching_offsets: Optional[keras.KerasTensor] = None
    
    def __init__(self, config: STULayerConfig, is_inference: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._config = config
        self._num_heads: int = config.num_heads
        self._embedding_dim: int = config.embedding_dim
        self._hidden_dim: int = config.hidden_dim
        self._attention_dim: int = config.attention_dim
        self._output_dropout_ratio: float = config.output_dropout_ratio
        self._target_aware: bool = config.target_aware
        self._causal: bool = config.causal
        self._max_attn_len: int = config.max_attn_len or 0
        self._attn_alpha: float = config.attn_alpha or 1.0 / (self._attention_dim**0.5)
        self._use_group_norm: bool = config.use_group_norm
        self._recompute_normed_x: bool = config.recompute_normed_x
        self._recompute_uvqk: bool = config.recompute_uvqk
        self._recompute_y: bool = config.recompute_y
        self._sort_by_length: bool = config.sort_by_length
        self._contextual_seq_len: int = config.contextual_seq_len
        self.reset_kv_cache()
    
    def build(self, input_shape):
        D_in = input_shape[-1]
        H = self._num_heads; A = self._attention_dim; V = self._hidden_dim
        output_dim_total = (V * 2 + A * 2) * H
        self._uvqk_weight = self.add_weight(shape=(D_in, output_dim_total), initializer='glorot_uniform', name='uvqk_weight')
        self._uvqk_beta = self.add_weight(shape=(output_dim_total,), initializer='zeros', name='uvqk_beta')
        self._input_norm_weight = self.add_weight(shape=(D_in,), initializer='ones', name='input_norm_weight')
        self._input_norm_bias = self.add_weight(shape=(D_in,), initializer='zeros', name='input_norm_bias')
        
        self._output_weight = self.add_weight(shape=(V * H, self._embedding_dim), initializer='glorot_uniform', name='output_weight')
        
        output_norm_shape: int = (V * H if not self._use_group_norm else H)
        self._output_norm_weight = self.add_weight(shape=(output_norm_shape,), initializer='ones', name='output_norm_weight')
        self._output_norm_bias = self.add_weight(shape=(output_norm_shape,), initializer='zeros', name='output_norm_bias')
        self.built = True

    def reset_kv_cache(self) -> None:
        self.k_cache = None; self.v_cache = None
        self.kv_caching_offsets = None; self.max_kv_caching_len = 0

    def update_kv_cache(
        self, max_seq_len: int, seq_offsets: keras.KerasTensor, k: Optional[keras.KerasTensor], v: Optional[keras.KerasTensor], max_kv_caching_len: int, kv_caching_lengths: Optional[keras.KerasTensor],
    ) -> None:
        # NOTE: Assumes _update_kv_cache is available
        self.k_cache, self.v_cache, self.max_kv_caching_len, self.kv_caching_offsets = _update_kv_cache(max_seq_len=max_seq_len, seq_offsets=seq_offsets, k=k, v=v, max_kv_caching_len=max_kv_caching_len, kv_caching_lengths=kv_caching_lengths, orig_k_cache=self.k_cache, orig_v_cache=self.v_cache, orig_max_kv_caching_len=self.max_kv_caching_len, orig_kv_caching_offsets=self.kv_caching_offsets)

    def construct_full_kv(self, delta_k: keras.KerasTensor, delta_v: keras.KerasTensor,) -> Tuple[keras.KerasTensor, keras.KerasTensor, int, keras.KerasTensor]:
        # NOTE: Assumes _construct_full_kv is available
        return _construct_full_kv(delta_k=delta_k, delta_v=delta_v, k_cache=fx_unwrap_optional_tensor(self.k_cache), v_cache=fx_unwrap_optional_tensor(self.v_cache), max_kv_caching_len=self.max_kv_caching_len, kv_caching_offsets=fx_unwrap_optional_tensor(self.kv_caching_offsets),)

    def call( # Standard Keras forward method
        self, x: keras.KerasTensor, x_lengths: keras.KerasTensor, x_offsets: keras.KerasTensor, max_seq_len: int, num_targets: keras.KerasTensor, max_kv_caching_len: int = 0, kv_caching_lengths: Optional[keras.KerasTensor] = None, training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        
        u, attn_output, k, v = keras_hstu_preprocess_and_attention( 
            x=x, norm_weight=self._input_norm_weight, norm_bias=self._input_norm_bias, norm_eps=1e-6,
            num_heads=self._num_heads, attn_dim=self._attention_dim, hidden_dim=self._hidden_dim,
            uvqk_weight=self._uvqk_weight, uvqk_bias=self._uvqk_beta,
            max_seq_len=max_seq_len, seq_offsets=x_offsets, attn_alpha=self._attn_alpha,
            causal=self._causal, num_targets=num_targets if self._target_aware else None,
            max_attn_len=self._max_attn_len, contextual_seq_len=self._contextual_seq_len,
            recompute_uvqk_in_backward=self._recompute_uvqk, recompute_normed_x_in_backward=self._recompute_normed_x,
            sort_by_length=self._sort_by_length, prefill=kv_caching_lengths is not None,
        )

        self.update_kv_cache(max_seq_len=max_seq_len, seq_offsets=x_offsets, k=k, v=v, max_kv_caching_len=max_kv_caching_len, kv_caching_lengths=kv_caching_lengths)

        return hstu_compute_output( 
            attn=attn_output, u=u, x=x, norm_weight=self._output_norm_weight, norm_bias=self._output_norm_bias,
            norm_eps=1e-6, dropout_ratio=self._output_dropout_ratio, output_weight=self._output_weight,
            group_norm=self._use_group_norm, num_heads=self._num_heads, linear_dim=self._hidden_dim,
            concat_ux=True, training=training, recompute_y_in_backward=self._recompute_y,
        )

    def cached_forward( # Called for token-by-token generation
        self, delta_x: keras.KerasTensor, num_targets: keras.KerasTensor, max_kv_caching_len: int = 0, kv_caching_lengths: Optional[keras.KerasTensor] = None, training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        
        delta_u, delta_q, delta_k, delta_v = hstu_compute_uqvk( 
            x=delta_x, norm_weight=self._input_norm_weight, norm_bias=self._input_norm_bias, norm_eps=1e-6,
            num_heads=self._num_heads, attn_dim=self._attention_dim, hidden_dim=self._hidden_dim,
            uvqk_weight=self._uvqk_weight, uvqk_bias=self._uvqk_beta,
        )
        
        A = self._attention_dim; V = self._hidden_dim; H = self._num_heads
        k_flat = ops.reshape(delta_k, [-1, H * A])
        v_flat = ops.reshape(delta_v, [-1, H * V])

        k_full, v_full, max_seq_len, seq_offsets = self.construct_full_kv(delta_k=k_flat, delta_v=v_flat)
        
        self.update_kv_cache(max_seq_len=max_seq_len, seq_offsets=seq_offsets, k=k_full, v=v_full, max_kv_caching_len=max_kv_caching_len, kv_caching_lengths=kv_caching_lengths)

        # Reshape K and V back to [L_full, H, D] for attention calculation
        k = ops.reshape(k_full, [-1, H, A])
        v = ops.reshape(v_full, [-1, H, V])

        
        delta_attn_output = delta_hstu_mha( 
            max_seq_len=max_seq_len, alpha=self._attn_alpha, delta_q=delta_q, k=k, v=v, seq_offsets=seq_offsets,
            num_targets=num_targets if self._target_aware else None, max_attn_len=self._max_attn_len,
            contextual_seq_len=self._contextual_seq_len,
        )
        
        delta_attn_output = ops.reshape(delta_attn_output, [-1, V * H])

       
        return hstu_compute_output( 
            attn=delta_attn_output, u=delta_u, x=delta_x, norm_weight=self._output_norm_weight, norm_bias=self._output_norm_bias,
            norm_eps=1e-6, dropout_ratio=self._output_dropout_ratio, output_weight=self._output_weight,
            group_norm=self._use_group_norm, num_heads=self._num_heads, linear_dim=self._hidden_dim,
            concat_ux=True, training=training, recompute_y_in_backward=self._recompute_y,
        )


class STUStack(layers.Layer):
    def __init__(self, stu_layers: List[STULayer], is_inference: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._stu_layers = stu_layers 
    
    def call(
        self, x: keras.KerasTensor, x_lengths: keras.KerasTensor, x_offsets: keras.KerasTensor, max_seq_len: int, num_targets: keras.KerasTensor, max_kv_caching_len: int = 0, kv_caching_lengths: Optional[keras.KerasTensor] = None, training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        for layer in self._stu_layers:
            x = layer(x=x, x_lengths=x_lengths, x_offsets=x_offsets, max_seq_len=max_seq_len, num_targets=num_targets, max_kv_caching_len=max_kv_caching_len, kv_caching_lengths=kv_caching_lengths, training=training)
        return x

    def cached_forward(
        self, delta_x: keras.KerasTensor, num_targets: keras.KerasTensor, max_kv_caching_len: int = 0, kv_caching_lengths: Optional[keras.KerasTensor] = None, training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        for layer in self._stu_layers:
            delta_x = layer.cached_forward(delta_x=delta_x, num_targets=num_targets, max_kv_caching_len=max_kv_caching_len, kv_caching_lengths=kv_caching_lengths, training=training)
        return delta_x
