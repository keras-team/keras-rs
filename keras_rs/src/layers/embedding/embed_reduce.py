from typing import Any

import keras
from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.utils.keras_utils import check_shapes_compatible

SUPPORTED_COMBINERS = ("mean", "sum", "sqrtn")


def _is_supported_sparse(x: types.Tensor) -> bool:
    """Determines if the input is a supported sparse tensor.

    NOTE: Currently only works for the TensorFlow and JAX backends.

    Args:
      x: Input tensor to check for sparsity.

    Returns:
      True if `x` is a supported sparse tensor.
    """
    if keras.backend.backend() == "tensorflow":
        import tensorflow as tf

        return isinstance(x, tf.SparseTensor)
    elif keras.backend.backend() == "jax":
        from jax.experimental import sparse as jax_sparse

        return isinstance(x, jax_sparse.BCOO) or isinstance(x, jax_sparse.BCSR)

    return False


def _sparse_ones_like(
    x: types.Tensor, dtype: types.DType | None = None
) -> types.Tensor:
    """Creates a tensor of ones with the same sparsity as the input.

    This differs from `keras.ops.ones_like`, which would create a dense
    tensor of ones.

    Args:
        x: Input sparse tensor.
        dtype: Optional dtype for the output tensor values.

    Returns:
        Sparse tensor of ones.

    Raises:
        ValueError for unsupported sparse input type and backend.
    """
    dtype = dtype or x.dtype
    if keras.backend.backend() == "tensorflow":
        import tensorflow as tf

        # Ensure shape is copied exactly for compatibility in graph mode.
        x_shape = x.shape
        y = tf.SparseTensor(
            x.indices, tf.ones_like(x.values, dtype=dtype), x.dense_shape
        )
        y.set_shape(x_shape)
        return y
    elif keras.backend.backend() == "jax":
        import jax.numpy as jnp
        from jax.experimental import sparse as jax_sparse

        if isinstance(x, jax_sparse.BCOO):
            return jax_sparse.BCOO(
                (jnp.ones_like(x.data, dtype=dtype), x.indices),
                shape=x.shape,
                indices_sorted=x.indices_sorted,
                unique_indices=x.unique_indices,
            )
        elif isinstance(x, jax_sparse.BCSR):
            return jax_sparse.BCSR(
                (jnp.ones_like(x.data, dtype=dtype), x.indices, x.indptr),
                shape=x.shape,
                indices_sorted=x.indices_sorted,
                unique_indices=x.unique_indices,
            )

    raise ValueError(
        f"Unsupported sparse input type '{x.__class__.__name__}' for backend "
        f"{keras.backend.backend()}."
    )


@keras_rs_export("keras_rs.layers.EmbedReduce")
class EmbedReduce(keras.layers.Embedding):
    """An embedding layer that reduces with a combiner.

    This layer embeds inputs and then applies a reduction to combine a set of
    embeddings into a single embedding. This is typically used to embed a
    sequence of items as a single embedding.

    If the inputs passed to `__call__` are 1D, no reduction is applied. If the
    inputs are 2D, dimension 1 is reduced using the combiner so that the result
    is of shape `(batch_size, output_dim`). Inputs of rank 3 and higher are not
    allowed. Weights can optionally be passed to the `__call__` method to
    apply weights to different samples before reduction.

    This layer supports sparse inputs and ragged inputs with backends that
    support them. The output after reduction is dense. For ragged inputs, the
    ragged dimension must be 1 as it is the dimension that is reduced.

    Args:
        input_dim: Integer. Size of the vocabulary, maximum integer index + 1.
        output_dim: Integer. Dimension of the dense embedding.
        embeddings_initializer: Initializer for the `embeddings` matrix (see
            `keras.initializers`).
        embeddings_regularizer: Regularizer function applied to the `embeddings`
            matrix (see `keras.regularizers`).
        embeddings_constraint: Constraint function applied to the `embeddings`
            matrix (see `keras.constraints`).
        mask_zero: Boolean, whether or not the input value 0 is a special
            "padding" value that should be masked out. This is useful when using
            recurrent layers which may take variable length input. If this is
            `True`, then all subsequent layers in the model need to support
            masking or an exception will be raised. If `mask_zero` is set to
            `True`, as a consequence, index 0 cannot be used in the vocabulary
            (`input_dim` should equal size of vocabulary + 1).
        weights: Optional floating-point matrix of size
            `(input_dim, output_dim)`. The initial embeddings values to use.
        combiner: Specifies how to reduce if there are multiple entries in a
            single row. Currently `mean`, `sqrtn` and `sum` are supported.
            `mean` is the default. `sqrtn` often achieves good accuracy, in
            particular with bag-of-words columns.
        **kwargs: Additional keyword arguments passed to `Embedding`.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embeddings_initializer: types.InitializerLike = "uniform",
        embeddings_regularizer: types.RegularizerLike | None = None,
        embeddings_constraint: types.ConstraintLike | None = None,
        mask_zero: bool = False,
        weights: types.Tensor = None,
        combiner: str = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_dim,
            output_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero,
            weights=weights,
            **kwargs,
        )
        if combiner not in SUPPORTED_COMBINERS:
            raise ValueError(
                f"Invalid `combiner`: '{combiner}', "
                f"use one of {', '.join(SUPPORTED_COMBINERS)}."
            )
        self.combiner = combiner

    def call(
        self,
        inputs: types.Tensor,
        weights: types.Tensor | None = None,
    ) -> types.Tensor:
        """Apply embedding and reduction.

        Args:
            inputs: 1D tensor to embed or 2D tensor to embed and reduce.
            weights: Optional tensor of weights to apply before reduction, which
               can be 1D or 2D and must match for the first dimension of
               `inputs` (1D case) or match the shape of `inputs` (2D case).

        Returns:
            A dense 2D tensor of shape `(batch_size, output_dim)`.
        """
        x = super().call(inputs)
        unreduced_rank = len(x.shape)

        # Check that weights has a compatible shape.
        if weights is not None:
            weights_rank = len(weights.shape)
            if weights_rank > unreduced_rank or not check_shapes_compatible(
                x.shape[0:weights_rank], weights.shape
            ):
                raise ValueError(
                    f"The shape of `weights`: {weights.shape} is not compatible"
                    f" with the shape of `inputs` after embedding: {x.shape}."
                )

        dtype = (
            x.dtype
            if weights is None
            else keras.backend.result_type(x.dtype, weights.dtype)
        )

        # When `weights` is `None`:
        # - For ragged inputs, after embedding, we get a ragged result that has
        #   a ragged dimension of 1, but when we do the "mean" or "sqrtn", we
        #   need to divide by the number of items in each row. However, there is
        #   no explicit cross backend API to get the row length. `ones_like`
        #   gives us a ragged tensor that is ragged in the same way as the
        #   inputs. When we do `ops.sum(weights, axis=-2)`, it gives us the
        #   number of items per row.
        # - For sparse inputs, after embedding, we get a dense tensor, not a
        #   sparse tensor. What it does for missing values is use embedding 0.
        #   These are bogus embedding and should be ignored. `ones_like` gives
        #   us a sparse tensor with the exact same missing values. Later, when
        #   we do `x = ops.multiply(x, weights)`, which masks the bogus values
        #   (note that `weights` has been densified beforehand). Additionally,
        #   when we do `ops.sum(weights, axis=-2)`, it gives us the number of
        #   items per row.
        #
        # When `unreduced_rank <= 2`, this means that the inputs where 1D and
        # dense, there is only one embedding per row, so there is no real
        # reduction is going on.
        # - For mean: result = weights * x / weights = x we don't need `weights`
        # - For sqrtn: result = weights * x / sqrt(square(weights)) = x we don't
        #   needs `weights`
        # - For sum however: `result = weights * x` we do need `weights`.
        # So for mean and sqrtn we don't need the weights, we use ones instead.
        # This is to avoid divisions by zero and improve the precision.
        if weights is None or (unreduced_rank <= 2 and self.combiner != "sum"):
            # Discard the weights if there were some and create a mask for
            # ragged and sparse tensors to mask the result correctly (sparse
            # only) and the apply the reduction correctly (ragged and sparse).
            if _is_supported_sparse(inputs):
                weights = _sparse_ones_like(inputs, dtype=dtype)
            else:
                weights = ops.ones_like(inputs, dtype=dtype)

        else:
            weights = ops.cast(weights, dtype)

        # When looking up using sparse indices, the result is dense but contains
        # values that should be ignored as all missing values use index 0. We
        # use `weights` as a mask, but it needs to be densified as
        # `expand_dims` and broadcasting a sparse tensor does not produce the
        # expected result.
        weights = ops.convert_to_tensor(weights, sparse=False)

        # Make weights and the unreduced embeddings have the same rank.
        weights_rank = len(weights.shape)
        if weights_rank < unreduced_rank:
            weights = ops.expand_dims(
                weights, axis=tuple(range(weights_rank, unreduced_rank))
            )

        # Note that `x` and `weights` are:
        # - ragged if `inputs` was ragged and `weights` was ragged or None
        # - dense otherwise (even if `inputs` and `weights` were sparse).
        x = ops.multiply(x, weights)

        if unreduced_rank <= 2:
            # No reduction is applied.
            return x

        # After this reduction, `x` is always dense as we reduce the ragged
        # dimension in the ragged case.
        x = ops.sum(x, axis=-2)

        # Apply the right divisor for the combiner.
        # Where we use `weights` in the divisor, we use
        # `ops.sum(weights, axis=-2)` which always makes it dense as we reduce
        # the ragged dimension in the ragged case.
        if self.combiner == "mean":
            return ops.divide_no_nan(x, ops.sum(weights, axis=-2))
        elif self.combiner == "sum":
            return x
        elif self.combiner == "sqrtn":
            return ops.divide_no_nan(
                x, ops.sqrt(ops.sum(ops.square(weights), axis=-2))
            )

    def compute_output_shape(
        self,
        input_shape: types.Shape,
        weights_shape: types.Shape | None = None,
    ) -> types.Shape:
        del weights_shape

        if len(input_shape) <= 1:
            # No reduction
            return (*input_shape, self.output_dim)
        else:
            # Reduce last dimension
            return (*input_shape[0:-1], self.output_dim)

    def compute_output_spec(
        self,
        inputs: keras.KerasTensor,
        weights: keras.KerasTensor | None = None,
    ) -> keras.KerasTensor:
        del weights

        output_shape = self.compute_output_shape(inputs.shape)
        return keras.KerasTensor(output_shape, dtype=self.compute_dtype)

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = super().get_config()

        config.update(
            {
                "combiner": self.combiner,
            }
        )

        return config
