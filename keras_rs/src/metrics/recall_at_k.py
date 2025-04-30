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


@keras_rs_export("keras_rs.metrics.RecallAtK")
class RecallAtK(RankingMetric):
    def compute_metric(
        self,
        y_true: types.Tensor,
        y_pred: types.Tensor,
        mask: types.Tensor,
        sample_weight: types.Tensor,
    ) -> types.Tensor:
        # Assume: `y_true = [0, 0, 1]`, `y_pred = [0.1, 0.9, 0.2]`.
        # `sorted_y_true = [0, 1, 0]` (sorted in descending order).
        (sorted_y_true,) = sort_by_scores(
            tensors_to_sort=[y_true],
            scores=y_pred,
            mask=mask,
            k=self.k,
            shuffle_ties=self.shuffle_ties,
            seed=self.seed_generator,
        )

        relevance = ops.cast(
            ops.greater_equal(
                sorted_y_true, ops.cast(1, dtype=sorted_y_true.dtype)
            ),
            dtype=y_pred.dtype,
        )
        overall_relevance = ops.cast(
            ops.greater_equal(y_true, ops.cast(1, dtype=y_true.dtype)),
            dtype=y_pred.dtype,
        )
        per_list_recall = ops.divide_no_nan(
            ops.sum(relevance, axis=1, keepdims=True),
            ops.sum(overall_relevance, axis=1, keepdims=True),
        )

        # Get weights.
        per_list_weights = get_list_weights(
            weights=sample_weight, relevance=overall_relevance
        )

        return per_list_recall, per_list_weights


concept_sentence = (
    "It measures the proportion of relevant items found in the top-k "
    "recommendations out of the total number of relevant items for a user"
)
relevance_type = "binary indicators (0 or 1) of relevance"
score_range_interpretation = (
    "Scores range from 0 to 1, with 1 indicating that all relevant items "
    "for the user were found within the top-k recommendations"
)
formula = """```
R@k(y, s) = sum_i I[rank(s_i) < k] y_i / sum_j y_j
```

where `y_i` is the relevance label (0/1) of the item ranked at position
`i`, `I[condition]` is 1 if the condition is met, otherwise 0."""
extra_args = ""
example = """
    >>> batch_size = 2
    >>> list_size = 5
    >>> labels = np.random.randint(0, 2, size=(batch_size, list_size))
    >>> scores = np.random.random(size=(batch_size, list_size))
    >>> metric = keras_rs.metrics.RecallAtK()(
    ...     y_true=labels, y_pred=scores
    ... )

    Mask certain elements (can be used for uneven inputs):

    >>> batch_size = 2
    >>> list_size = 5
    >>> labels = np.random.randint(0, 2, size=(batch_size, list_size))
    >>> scores = np.random.random(size=(batch_size, list_size))
    >>> mask = np.random.randint(0, 2, size=(batch_size, list_size), dtype=bool)
    >>> metric = keras_rs.metrics.RecallAtK()(
    ...     y_true={"labels": labels, "mask": mask}, y_pred=scores
    ... )
"""

RecallAtK.__doc__ = format_docstring(
    ranking_metric_subclass_doc_string,
    width=80,
    metric_name="Recall@k",
    metric_abbreviation="R@k",
    concept_sentence=concept_sentence,
    relevance_type=relevance_type,
    score_range_interpretation=score_range_interpretation,
    formula=formula,
) + ranking_metric_subclass_doc_string_post_desc.format(
    extra_args=extra_args, example=example
)
