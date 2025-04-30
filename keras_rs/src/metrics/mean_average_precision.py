from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.metrics.ranking_metric import RankingMetric
from keras_rs.src.metrics.ranking_metric import (
    ranking_metric_subclass_doc_string,
)
from keras_rs.src.metrics.ranking_metric import (
    ranking_metric_subclass_doc_string_post_desc,
)
from keras_rs.src.metrics.ranking_metrics_utils import get_list_weights
from keras_rs.src.metrics.ranking_metrics_utils import sort_by_scores
from keras_rs.src.utils.doc_string_utils import format_docstring


@keras_rs_export("keras_rs.metrics.MeanAveragePrecision")
class MeanAveragePrecision(RankingMetric):
    def compute_metric(
        self,
        y_true: types.Tensor,
        y_pred: types.Tensor,
        mask: types.Tensor,
        sample_weight: types.Tensor,
    ) -> types.Tensor:
        relevance = ops.cast(
            ops.greater_equal(y_true, ops.cast(1, dtype=y_true.dtype)),
            dtype=y_pred.dtype,
        )
        sorted_relevance, sorted_weights = sort_by_scores(
            tensors_to_sort=[relevance, sample_weight],
            scores=y_pred,
            mask=mask,
            k=self.k,
            shuffle_ties=self.shuffle_ties,
            seed=self.seed_generator,
        )
        per_list_relevant_counts = ops.cumsum(sorted_relevance, axis=1)
        per_list_cutoffs = ops.cumsum(ops.ones_like(sorted_relevance), axis=1)
        per_list_precisions = ops.divide_no_nan(
            per_list_relevant_counts, per_list_cutoffs
        )

        total_precision = ops.sum(
            ops.multiply(
                per_list_precisions,
                ops.multiply(sorted_weights, sorted_relevance),
            ),
            axis=1,
            keepdims=True,
        )

        # Compute the total relevance.
        total_relevance = ops.sum(
            ops.multiply(sample_weight, relevance), axis=1, keepdims=True
        )

        per_list_map = ops.divide_no_nan(total_precision, total_relevance)

        per_list_weights = get_list_weights(sample_weight, relevance)

        return per_list_map, per_list_weights


concept_sentence = (
    "It calculates the average of precision values computed after each "
    "relevant item present in the ranked list"
)
relevance_type = "binary indicators (0 or 1) of relevance"
score_range_interpretation = (
    "Scores range from 0 to 1, with higher values indicating that relevant "
    "items are generally positioned higher in the ranking"
)

formula = """
The formula for average precision is defined below. MAP is the mean over average
precision computed for each list.

```
AP(y, s) = sum_j (P@j(y, s) * rel(j)) / sum_i y_i
rel(j) = y_i if rank(s_i) = j
```

where:
- `j` represents the rank position (starting from 1).
- `sum_j` indicates a summation over all ranks `j` from 1 up to the list
  size (or `k`).
- `P@j(y, s)` denotes the Precision at rank `j`, calculated as the
  number of relevant items found within the top `j` positions divided by `j`.
- `rel(j)` represents the relevance of the item specifically at rank
  `j`. `rel(j)` is 1 if the item at rank `j` is relevant, and 0 otherwise.
- `y_i` is the true relevance label of the original item `i` before ranking.
- `rank(s_i)` is the rank position assigned to item `i` based on its score
  `s_i`.
- `sum_i y_i` calculates the total number of relevant items in the original
  list `y`."""
extra_args = ""
example = """
    >>> batch_size = 2
    >>> list_size = 5
    >>> labels = np.random.randint(0, 2, size=(batch_size, list_size))
    >>> scores = np.random.random(size=(batch_size, list_size))
    >>> metric = keras_rs.metrics.MeanAveragePrecision()(
    ...     y_true=labels, y_pred=scores
    ... )

    Mask certain elements (can be used for uneven inputs):

    >>> batch_size = 2
    >>> list_size = 5
    >>> labels = np.random.randint(0, 2, size=(batch_size, list_size))
    >>> scores = np.random.random(size=(batch_size, list_size))
    >>> mask = np.random.randint(0, 2, size=(batch_size, list_size), dtype=bool)
    >>> metric = keras_rs.metrics.MeanAveragePrecision()(
    ...     y_true={"labels": labels, "mask": mask}, y_pred=scores
    ... )
"""

MeanAveragePrecision.__doc__ = format_docstring(
    ranking_metric_subclass_doc_string,
    width=80,
    metric_name="Mean Average Precision",
    metric_abbreviation="MAP",
    concept_sentence=concept_sentence,
    relevance_type=relevance_type,
    score_range_interpretation=score_range_interpretation,
    formula=formula,
) + ranking_metric_subclass_doc_string_post_desc.format(
    extra_args=extra_args, example=example
)
