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


@keras_rs_export("keras_rs.metrics.PrecisionAtK")
class PrecisionAtK(RankingMetric):
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

        # We consider only binary relevance here, anything above 1 is treated
        # as 1. `relevance = [0., 1., 0.]`.
        relevance = ops.cast(
            ops.greater_equal(
                sorted_y_true, ops.cast(1, dtype=sorted_y_true.dtype)
            ),
            dtype=y_pred.dtype,
        )
        list_length = ops.shape(sorted_y_true)[1]
        # TODO: We do not do this for MRR, and the other metrics. Do we need to
        # do this there too?
        valid_list_length = ops.minimum(
            list_length,
            ops.sum(ops.cast(mask, dtype="int32"), axis=1, keepdims=True),
        )

        per_list_precision = ops.divide_no_nan(
            ops.sum(relevance, axis=1, keepdims=True),
            ops.cast(valid_list_length, dtype=y_pred.dtype),
        )

        # Get weights.
        overall_relevance = ops.cast(
            ops.greater_equal(y_true, ops.cast(1, dtype=y_true.dtype)),
            dtype=y_pred.dtype,
        )
        per_list_weights = get_list_weights(
            weights=sample_weight, relevance=overall_relevance
        )

        return per_list_precision, per_list_weights


concept_sentence = (
    "It measures the proportion of relevant items among the top-k "
    "recommendations"
)
relevance_type = "binary indicators (0 or 1) of relevance"
score_range_interpretation = (
    "Scores range from 0 to 1, with 1 indicating all top-k items were relevant"
)
formula = """```
P@k(y, s) = 1/k sum_i I[rank(s_i) < k] y_i
```

where `y_i` is the relevance label (0/1) of the item ranked at position
`i`, and `I[condition]` is 1 if the condition is met, otherwise 0."""
extra_args = ""
example = """
    >>> batch_size = 2
    >>> list_size = 5
    >>> labels = np.random.randint(0, 2, size=(batch_size, list_size))
    >>> scores = np.random.random(size=(batch_size, list_size))
    >>> metric = keras_rs.metrics.PrecisionAtK()(
    ...     y_true=labels, y_pred=scores
    ... )

    Mask certain elements (can be used for uneven inputs):

    >>> batch_size = 2
    >>> list_size = 5
    >>> labels = np.random.randint(0, 2, size=(batch_size, list_size))
    >>> scores = np.random.random(size=(batch_size, list_size))
    >>> mask = np.random.randint(0, 2, size=(batch_size, list_size), dtype=bool)
    >>> metric = keras_rs.metrics.PrecisionAtK()(
    ...     y_true={"labels": labels, "mask": mask}, y_pred=scores
    ... )
"""

PrecisionAtK.__doc__ = format_docstring(
    ranking_metric_subclass_doc_string,
    width=80,
    metric_name="Precision@k",
    metric_abbreviation="P@k",
    concept_sentence=concept_sentence,
    relevance_type=relevance_type,
    score_range_interpretation=score_range_interpretation,
    formula=formula,
) + ranking_metric_subclass_doc_string_post_desc.format(
    extra_args=extra_args, example=example
)
