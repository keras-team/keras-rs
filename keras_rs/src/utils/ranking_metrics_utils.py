from typing import Callable, Optional

import keras
from keras import ops

from keras_rs.src import types


def get_shuffled_indices(
    shape: types.TensorShape,
    mask: Optional[types.Tensor] = None,
    shuffle_ties: bool = True,
    seed: Optional[int] = None,
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
    mask: Optional[types.Tensor] = None,
    k: Optional[int] = None,
    seed: Optional[int] = None,
    shuffle_ties: bool = True,
) -> types.Tensor:
    """
    Utility function for sorting tensors by scores.

    Args:
        tensors_to_sort. list of tensors. All tensors are of shape
            `(batch_size, list_size)`. These tensors are sorted based on
            `scores`.
        scores: tensor. Of shape `(batch_size, list_size)`. The scores to sort
            by.
        k: int. The number of top-ranked items to consider (the 'k' in 'top-k').
            If `None`, `list_size` is used.
        seed: int. Seed for shuffling.
        shuffle_ties: bool. Whether to randomly shuffle scores before sorting.
            This is done to break ties.

    Returns:
        List of sorted tensors (`tensors_to_sort`), sorted using `scores`.
    """
    # TODO: Consider exposing `shuffle_ties` to the user.
    # TODO: Figure out `seed`. How do we propagate it down here from ranking
    # metric?
    max_possible_k = ops.shape(scores)[1]
    if k is None:
        k = max_possible_k
    else:
        k = ops.minimum(k, max_possible_k)

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

    The per-list weights are computed as:
    ```
    per_list_weights = sum(weights * relevance) / sum(relevance).
    ```

    For a list with sum(relevance) = 0, we set a default weight as the following
    average weight while all the lists with sum(weights) = 0 are ignored:

    ```
    sum(per_list_weights) / num(sum(relevance) != 0 && sum(weights) != 0)
    ```

    When all the lists have sum(relevance) == 0, we set the average weight to
    1.0.

    This computation takes care of the following cases:
    - When all the weights are 1.0, the per-list weights will be 1.0
      everywhere, even for lists without any relevant examples because
      `sum(per_list_weights) ==  num(sum(relevance) != 0)`. This handles the
      standard ranking metrics where the weights are all
      1.0.
    - When every list has a nonzero weight, the default weight is not used.
      This handles the unbiased metrics well.
    - For the mixture of the above 2 scenario, the weights for lists with
      nonzero relevance and nonzero weights is proportional to

      ```
      per_list_weights / sum(per_list_weights) *
      num(sum(relevance) != 0) / num(lists)
      ```

      The rest have weights 1.0 / num(lists).

    Args:
        weights:  The weights `Tensor` of shape [batch_size, list_size].
        relevance:  The relevance `Tensor` of shape [batch_size, list_size].

    Returns:
        A tensor of shape [batch_size, 1], containing the per-list weights.
    """
    # TODO: Pretty nasty function. Check if we can simplify it at a later point.

    # Calculate if the sum of weights per list is greater than 0.0.
    nonzero_weights = ops.greater(ops.sum(weights, axis=1, keepdims=True), 0.0)
    # Calculate the sum of relevance per list
    per_list_relevance = ops.sum(relevance, axis=1, keepdims=True)

    # Identify lists where both weights and relevance sums are non-zero.
    nonzero_relevance_condition = ops.greater(per_list_relevance, 0.0)
    nonzero_relevance = ops.where(
        nonzero_weights,
        ops.cast(nonzero_relevance_condition, "float32"),
        ops.zeros_like(per_list_relevance),
    )
    # Count the number of lists with non-zero relevance (and implicitly
    # non-zero weights).
    nonzero_relevance_count = ops.sum(nonzero_relevance, axis=0, keepdims=True)

    # Calculate the per-list weights using the core formula
    # Numerator: sum(weights * relevance) per list
    numerator = ops.sum(weights * relevance, axis=1, keepdims=True)
    # Denominator: per_list_relevance = sum(relevance) per list
    per_list_weights = ops.divide_no_nan(numerator, per_list_relevance)

    # Calculate the sum of the initially computed per-list weights (where
    # relevance > 0)
    sum_weights = ops.sum(per_list_weights, axis=0, keepdims=True)

    # Calculate the average weight to use as default for lists with zero
    # relevance but non-zero weights. If no lists have non-zero relevance,
    # default to 1.0.
    avg_weight = ops.where(
        ops.greater(nonzero_relevance_count, 0.0),
        ops.divide_no_nan(sum_weights, nonzero_relevance_count),
        ops.ones_like(nonzero_relevance_count),
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
            ops.ones_like(per_list_weights) * avg_weight,
        ),
        ops.zeros_like(per_list_weights),
    )

    return final_weights


def default_gain_fn(label: types.Tensor) -> types.Tensor:
    return ops.subtract(ops.power(2.0, label), 1.0)


def default_rank_discount_fn(rank: types.Tensor) -> types.Tensor:
    return ops.divide(ops.log(2.0), ops.log1p(rank))


def compute_dcg(
    y_true: types.Tensor,
    sample_weight: types.Tensor,
    gain_fn: Callable[[types.Tensor], types.Tensor] = default_gain_fn,
    rank_discount_fn: Callable[
        [types.Tensor], types.Tensor
    ] = default_rank_discount_fn,
) -> types.Tensor:
    list_size = ops.shape(y_true)[1]
    positions = ops.arange(1, list_size + 1, dtype="float32")
    gain = gain_fn(y_true)
    discount = rank_discount_fn(positions)

    return ops.sum(
        ops.multiply(sample_weight, ops.multiply(gain, discount)),
        axis=1,
        keepdims=True,
    )
