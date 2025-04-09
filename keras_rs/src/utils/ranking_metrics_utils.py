from typing import Optional

import keras
from keras import ops

from keras_rs.src import types


def get_shuffled_and_masked_indices(
    shape: types.TensorShape,
    mask: Optional[types.Tensor] = None,
    shuffle_ties: bool = True,
    seed: Optional[int] = None,
) -> types.Tensor:
    # If shuffle_ties is True, generate random values. Otherwise, generate
    # zeros so that we get [0, 1, 2, ...] as indices on doing `argsort`.
    if shuffle_ties:
        shuffle_values = keras.random.uniform(shape, seed=seed, dtype="float32")
    else:
        shuffle_values = ops.zeros(shape, dtype="float32")

    # When `mask = False`, increase value by 1 so that those indices are placed
    # last. Note that `shuffle_values` lies in the range `[0, 1)`, so adding by
    # 1 works out.
    if mask is not None:
        shuffle_values = ops.where(
            mask,
            shuffle_values,
            ops.add(shuffle_values, ops.cast(1, dtype="float32")),
        )

    shuffled_indices = ops.argsort(shuffle_values)
    return shuffled_indices


def sort_by_scores(
    y_true: types.Tensor,
    y_pred: types.Tensor,
    mask: Optional[types.Tensor] = None,
    k: Optional[int] = None,
    seed: Optional[int] = None,
    shuffle_ties: bool = True,
) -> types.Tensor:
    max_possible_k = ops.shape(y_pred)[1]
    if k is None:
        k = max_possible_k
    else:
        k = ops.minimum(k, max_possible_k)

    # Set values corresponding to `mask = False` to the minimum value so that
    # they are last when we sort.
    if mask is not None:
        y_pred = ops.where(mask, y_pred, ops.minimum(y_pred))

    # Shuffle ties randomly, and push masked values to the beginning.
    shuffled_indices = None
    if shuffle_ties or mask is not None:
        shuffled_indices = get_shuffled_and_masked_indices(
            ops.shape(y_pred),
            mask=mask,
            shuffle_ties=True,
            seed=seed,
        )
        y_pred = ops.take_along_axis(y_pred, shuffled_indices, axis=1)

    _, indices = ops.top_k(y_pred, k=k, sorted=True)
    if shuffled_indices is not None:
        indices = ops.take_along_axis(shuffled_indices, indices, axis=1)

    sorted_y_true = ops.take_along_axis(y_true, indices, axis=1)
    return sorted_y_true


def get_list_weights(
    weights: types.Tensor, relevance: types.Tensor
) -> types.Tensor:
    """Computes per list weight from per example weight using keras.ops.

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

    Such a computation is good for the following scenarios:
        - When all the weights are 1.0, the per-list weights will be 1.0
          everywhere, even for lists without any relevant examples because
          `sum(per_list_weights) ==  num(sum(relevance) != 0)`.
          This handles the standard ranking metrics where the weights are all
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
        The per list `Tensor` of shape [batch_size, 1]
    """
    # Calculate if the sum of weights per list is greater than 0.0.
    nonzero_weights = ops.greater(
        ops.sum(weights, axis=1, keepdims=True), 0.0
    )  # Shape: [batch_size, 1], dtype: bool

    # Calculate the sum of relevance per list
    per_list_relevance = ops.sum(
        relevance, axis=1, keepdims=True
    )  # Shape: [batch_size, 1]

    # Identify lists where both weights and relevance sums are non-zero.
    # Only consider lists with non-zero weights for this check.
    nonzero_relevance_condition = ops.greater(
        per_list_relevance, 0.0
    )  # Shape: [batch_size, 1], dtype: bool
    nonzero_relevance = ops.where(
        nonzero_weights,
        ops.cast(
            nonzero_relevance_condition, "float32"
        ),  # Cast boolean to float (1.0 or 0.0)
        ops.zeros_like(per_list_relevance),  # Use 0.0 if weights are zero
    )  # Shape: [batch_size, 1], float32

    # Count the number of lists with non-zero relevance (and implicitly
    # non-zero weights).
    nonzero_relevance_count = ops.sum(
        nonzero_relevance, axis=0, keepdims=True
    )  # Shape: [1, 1]

    # Calculate the per-list weights using the core formula
    # Numerator: sum(weights * relevance) per list
    numerator = ops.sum(
        weights * relevance, axis=1, keepdims=True
    )  # Shape: [batch_size, 1]
    # Denominator: per_list_relevance = sum(relevance) per list
    per_list_weights = ops.divide_no_nan(
        numerator, per_list_relevance
    )  # Shape: [batch_size, 1]

    # Calculate the sum of the initially computed per-list weights (where
    # relevance > 0)
    sum_weights = ops.sum(
        per_list_weights, axis=0, keepdims=True
    )  # Shape: [1, 1]

    # Calculate the average weight to use as default for lists with zero
    # relevance but non-zero weights. If no lists have non-zero relevance,
    # default to 1.0.
    avg_weight = ops.where(
        ops.greater(nonzero_relevance_count, 0.0),
        ops.divide_no_nan(
            sum_weights, nonzero_relevance_count
        ),  # avg = sum / count
        ops.ones_like(nonzero_relevance_count),  # default to 1.0
    )  # Shape: [1, 1]

    # Final assignment of weights based on conditions:
    # 1. If sum(weights) == 0 for a list, the final weight is 0.
    # 2. If sum(weights) > 0 AND sum(relevance) > 0, use the calculated
    # `per_list_weights`.
    # 3. If `sum(weights) > 0` AND `sum(relevance) == 0`, use the calculated
    # `avg_weight`.
    final_weights = ops.where(
        nonzero_weights,  # Condition 1 check (inverted)
        ops.where(
            nonzero_relevance_condition,  # Condition 2/3 check
            per_list_weights,  # Use calculated weight if relevance > 0
            ops.ones_like(per_list_weights)
            * avg_weight,  # Use avg_weight if relevance == 0
        ),
        ops.zeros_like(per_list_weights),  # Use 0 if weights sum is 0
    )  # Shape: [batch_size, 1]

    return final_weights
