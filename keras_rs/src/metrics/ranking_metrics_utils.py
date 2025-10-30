from typing import Callable

import keras
from keras import ops

from keras_rs.src import types


def get_shuffled_indices(
    shape: types.Shape,
    mask: types.Tensor | None = None,
    shuffle_ties: bool = True,
    seed: int | keras.random.SeedGenerator | None = None,
) -> types.Tensor:
    """Utility function for getting shuffled indices, with masked indices
    pushed to the end.

    Args:
        shape: tuple. The shape of the tensor for which to generate
            shuffled indices.
        mask: An optional boolean tensor with the same shape as `shape`.
            If provided, elements where `mask` is `False` will be placed
            at the end of the sorted indices. Defaults to `None` (no masking).
        shuffle_ties: Boolean indicating how to handle ties if multiple elements
            have the same sorting value (randomly when `shuffle_ties` is True
            otherwise, order is preserved).
        seed: Optional integer seed for the random number generator used when
            `shuffle_ties` is True. Ensures reproducibility. Defaults to None.

    Returns:
        A tensor of shape `shape` containing shuffled indices.
    """
    # If `shuffle_ties` is True, generate random values. Otherwise, generate
    # zeros so that we get `[0, 1, 2, ...]` as indices on doing `argsort`.
    if shuffle_ties:
        shuffle_values = keras.random.uniform(shape, seed=seed, dtype="float32")
    else:
        shuffle_values = ops.zeros(shape, dtype="float32")

    # When `mask = False`, increase value by 1 so that those indices are placed
    # at the end. Note that `shuffle_values` lies in the range `[0, 1)`, so
    # adding by 1 works out.
    if mask is not None:
        shuffle_values = ops.where(
            mask,
            shuffle_values,
            ops.add(shuffle_values, ops.cast(1, dtype="float32")),
        )

    shuffled_indices = ops.argsort(shuffle_values)
    return shuffled_indices


def sort_by_scores(
    tensors_to_sort: list[types.Tensor],
    scores: types.Tensor,
    mask: types.Tensor | None = None,
    k: int | None = None,
    shuffle_ties: bool = True,
    seed: int | keras.random.SeedGenerator | None = None,
) -> types.Tensor:
    """
    Utility function for sorting tensors by scores.

    Args:
        tensors_to_sort: list of tensors. All tensors are of shape
            `(batch_size, list_size)`. These tensors are sorted based on
            `scores`.
        scores: tensor. Of shape `(batch_size, list_size)`. The scores to sort
            by.
        k: int. The number of top-ranked items to consider (the 'k' in 'top-k').
            If `None`, `list_size` is used.
        shuffle_ties: bool. Whether to randomly shuffle scores before sorting.
            This is done to break ties.
        seed: int. Seed for shuffling.

    Returns:
        List of sorted tensors (`tensors_to_sort`), sorted using `scores`.
    """
    max_possible_k = ops.shape(scores)[1]
    if k is None:
        k = max_possible_k
    elif isinstance(max_possible_k, int):
        k = min(k, max_possible_k)
    else:
        k = ops.minimum(k, max_possible_k)

    # --- Work around for PyTorch instability ---
    # Torch's `topk` is not stable with `sorted=True`, unlike JAX and TF.
    # See:
    #   - https://github.com/pytorch/pytorch/issues/27542
    #   - https://github.com/pytorch/pytorch/issues/88227
    #
    # This small "stable offset" ensures deterministic tie-breaking for
    # equal scores. We can remove this workaround once PyTorch adds a
    # `stable=True` flag for topk.

    if keras.backend.backend() == "torch" and not shuffle_ties:
        list_size = ops.shape(scores)[1]
        indices = ops.arange(list_size)
        indices = ops.expand_dims(indices, axis=0)
        indices = ops.broadcast_to(indices, ops.shape(scores))
        stable_offset = ops.cast(indices, scores.dtype) * 1e-6
        scores = ops.subtract(scores, stable_offset)
    # --- End FIX ---

    # Shuffle ties randomly, and push masked values to the beginning.
    shuffled_indices = None
    if shuffle_ties or mask is not None:
        shuffled_indices = get_shuffled_indices(
            ops.shape(scores),
            mask=mask,
            shuffle_ties=True,
            seed=seed,
        )
        scores = ops.take_along_axis(scores, shuffled_indices, axis=1)

    # Get top-k indices.
    _, indices = ops.top_k(scores, k=k, sorted=True)

    # If we shuffled our `scores` tensor, we need to get the correct indices
    # by indexing into `shuffled_indices`.
    if shuffled_indices is not None:
        indices = ops.take_along_axis(shuffled_indices, indices, axis=1)

    return [
        ops.take_along_axis(tensor_to_sort, indices, axis=1)
        for tensor_to_sort in tensors_to_sort
    ]


def get_list_weights(
    weights: types.Tensor, relevance: types.Tensor
) -> types.Tensor:
    """Computes per-list weights from provided sample weights.

    Per-list weights are calculated as follows:
    ```
    per_list_weights = sum(weights * relevance) / sum(relevance).
    ```

    For lists where the sum of relevance is 0, a default weight is assigned:
    ```
    sum(per_list_weights) / num(sum(relevance) != 0 AND sum(weights) != 0)
    ```

    If all lists have a sum of relevance equal to 0, the default weight is 1.0.

    As a result of the above computation, this function takes care of the
    following cases:
    - **Uniform Weights:** When all input weights are 1.0, all per-list weights
      will be 1.0, even for lists with no relevant examples. This aligns with
      standard ranking metrics.
    - **Non-zero Weights per List:** If every list has at least one non-zero
      weight, the default weight mechanism is not utilized, which is suitable
      for unbiased metrics.
    - **Mixed Scenarios:** For cases with a mix of lists having zero and
      non-zero relevance and weights, the weights for lists with non-zero
      relevance and weights are proportional to:

      ```
      per_list_weights / sum(per_list_weights) *
      num(sum(relevance) != 0) / num(lists)
      ```

      The rest have weights `1.0 / num(lists)`.

    Args:
        weights: tensor. Weights tensor of shape `(batch_size, list_size)`.
        relevance: tensor. The relevance `Tensor` of shape
            `(batch_size, list_size)`.

    Returns:
        A tensor of shape `(batch_size, 1)`, containing the per-list weights.
    """
    # Calculate if the sum of weights per list is greater than 0.0.
    nonzero_weights = ops.greater(ops.sum(weights, axis=1, keepdims=True), 0.0)
    # Calculate the sum of relevance per list
    per_list_relevance = ops.sum(relevance, axis=1, keepdims=True)
    # Calculate if the sum of relevance per list is greater than 0.0
    nonzero_relevance_condition = ops.greater(per_list_relevance, 0.0)
    # Identify lists where both weights and relevance sums are non-zero.
    nonzero_relevance = ops.cast(
        ops.logical_and(nonzero_weights, nonzero_relevance_condition),
        dtype=weights.dtype,
    )
    # Count the number of lists with non-zero relevance and non-zero weights.
    nonzero_relevance_count = ops.sum(nonzero_relevance, axis=0, keepdims=True)

    # Calculate the per-list weights using the core formula.
    # Numerator: `sum(weights * relevance)` per list
    numerator = ops.sum(ops.multiply(weights, relevance), axis=1, keepdims=True)
    # Denominator: per_list_relevance = sum(relevance) per list
    per_list_weights = ops.divide_no_nan(numerator, per_list_relevance)

    # Calculate the sum of the computed per-list weights.
    sum_weights = ops.sum(per_list_weights, axis=0, keepdims=True)

    # Calculate the average weight to use as default for lists with zero
    # relevance but non-zero weights. If no lists have non-zero relevance,
    # default to 1.0.
    avg_weight = ops.where(
        ops.greater(nonzero_relevance_count, 0.0),
        ops.divide(sum_weights, nonzero_relevance_count),
        ops.cast(1, dtype=sum_weights.dtype),
    )

    # Final assignment of weights based on conditions:
    # 1. If sum(weights) == 0 for a list, the final weight is 0.
    # 2. If sum(weights) > 0 AND sum(relevance) > 0, use the calculated
    # `per_list_weights`.
    # 3. If `sum(weights) > 0` AND `sum(relevance) == 0`, use the calculated
    # `avg_weight`.
    final_weights = ops.where(
        nonzero_weights,
        ops.where(
            nonzero_relevance_condition,
            per_list_weights,
            avg_weight,
        ),
        ops.cast(0, dtype=per_list_weights.dtype),
    )

    return final_weights


@keras.saving.register_keras_serializable()  # type: ignore[misc]
def default_gain_fn(label: types.Tensor) -> types.Tensor:
    return ops.subtract(ops.power(2.0, label), 1.0)


@keras.saving.register_keras_serializable()  # type: ignore[misc]
def default_rank_discount_fn(rank: types.Tensor) -> types.Tensor:
    return ops.divide(
        ops.cast(1, dtype=rank.dtype),
        ops.log2(ops.add(ops.cast(1, dtype=rank.dtype), rank)),
    )


def compute_dcg(
    y_true: types.Tensor,
    sample_weight: types.Tensor,
    gain_fn: Callable[[types.Tensor], types.Tensor] = default_gain_fn,
    rank_discount_fn: Callable[
        [types.Tensor], types.Tensor
    ] = default_rank_discount_fn,
) -> types.Tensor:
    list_size = ops.shape(y_true)[1]
    positions = ops.arange(1, list_size + 1, dtype=y_true.dtype)
    gain = gain_fn(y_true)
    discount = rank_discount_fn(positions)

    return ops.sum(
        ops.multiply(sample_weight, ops.multiply(gain, discount)),
        axis=1,
        keepdims=True,
    )
