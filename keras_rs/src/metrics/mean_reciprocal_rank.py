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


@keras_rs_export("keras_rs.metrics.MeanReciprocalRank")
class MeanReciprocalRank(RankingMetric):
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

        # This will depend on `k`, i.e., it will not always be the same as
        # `len(y_true)`.
        list_length = ops.shape(sorted_y_true)[1]

        # We consider only binary relevance here, anything above 1 is treated
        # as 1. `relevance = [0., 1., 0.]`.
        relevance = ops.cast(
            ops.greater_equal(
                sorted_y_true, ops.cast(1, dtype=sorted_y_true.dtype)
            ),
            dtype=y_pred.dtype,
        )

        # `reciprocal_rank = [1, 0.5, 0.33]`
        reciprocal_rank = ops.divide(
            ops.cast(1, dtype=y_pred.dtype),
            ops.arange(1, list_length + 1, dtype=y_pred.dtype),
        )

        # `mrr` should be of shape `(batch_size, 1)`.
        # `mrr = amax([0., 0.5, 0.]) = 0.5`
        mrr = ops.amax(
            ops.multiply(relevance, reciprocal_rank),
            axis=1,
            keepdims=True,
        )

        # Get weights.
        overall_relevance = ops.cast(
            ops.greater_equal(y_true, ops.cast(1, dtype=y_true.dtype)),
            dtype=y_pred.dtype,
        )
        per_list_weights = get_list_weights(
            weights=sample_weight, relevance=overall_relevance
        )

        return mrr, per_list_weights


concept_sentence = (
    "It focuses on the rank position of the single highest-scoring relevant "
    "item"
)
relevance_type = "binary indicators (0 or 1) of relevance"
score_range_interpretation = (
    "Scores range from 0 to 1, with 1 indicating the first relevant item was "
    "always ranked first"
)
formula = """```
MRR(y, s) = max_{i} y_{i} / rank(s_{i})
```"""
extra_args = ""
example = """
    >>> batch_size = 2
    >>> list_size = 5
    >>> labels = np.random.randint(0, 2, size=(batch_size, list_size))
    >>> scores = np.random.random(size=(batch_size, list_size))
    >>> metric = keras_rs.metrics.MeanReciprocalRank()(
    ...     y_true=labels, y_pred=scores
    ... )

    Mask certain elements (can be used for uneven inputs):

    >>> batch_size = 2
    >>> list_size = 5
    >>> labels = np.random.randint(0, 2, size=(batch_size, list_size))
    >>> scores = np.random.random(size=(batch_size, list_size))
    >>> mask = np.random.randint(0, 2, size=(batch_size, list_size), dtype=bool)
    >>> metric = keras_rs.metrics.MeanReciprocalRank()(
    ...     y_true={"labels": labels, "mask": mask}, y_pred=scores
    ... )
"""

MeanReciprocalRank.__doc__ = format_docstring(
    ranking_metric_subclass_doc_string,
    width=80,
    metric_name="Mean Reciprocal Rank",
    metric_abbreviation="MRR",
    concept_sentence=concept_sentence,
    relevance_type=relevance_type,
    score_range_interpretation=score_range_interpretation,
    formula=formula,
) + ranking_metric_subclass_doc_string_post_desc.format(
    extra_args=extra_args, example=example
)
