import math

import keras
from absl.testing import parameterized
from keras import ops
from keras.metrics import deserialize
from keras.metrics import serialize

from keras_rs.src import testing
from keras_rs.src.metrics.n_dcg import nDCG


def _compute_dcg(labels, ranks):
    val = 0.0
    for label, rank in zip(labels, ranks):
        val += (math.pow(2, label) - 1) / math.log2(rank + 1)
    return val


class nDCGTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        # === Unbatched Inputs ===
        # Binary relevance example
        self.y_true_unbatched = ops.array([0, 0, 1, 0], dtype="float32")
        self.y_pred_unbatched_perfect = ops.array(
            [0.1, 0.2, 0.9, 0.3], dtype="float32"
        )
        self.y_pred_unbatched_good = ops.array(
            [0.8, 0.1, 0.7, 0.2], dtype="float32"
        )
        self.y_pred_unbatched_bad = ops.array(
            [0.4, 0.3, 0.2, 0.1], dtype="float32"
        )

        # No relevant items
        self.y_true_unbatched_none = ops.array([0, 0, 0, 0], dtype="float32")

        # Graded relevance example
        self.y_true_unbatched_graded = ops.array([1, 0, 3, 2], dtype="float32")
        self.y_pred_unbatched_graded_perfect = ops.array(
            [0.1, 0.8, 0.9, 0.7], dtype="float32"
        )
        self.y_pred_unbatched_graded_mixed = ops.array(
            [0.9, 0.1, 0.7, 0.8], dtype="float32"
        )

        # === Batched Inputs ===
        self.y_true_batched = ops.array(
            [
                [0, 0, 1, 0],
                [1, 0, 3, 2],
                [0, 0, 0, 0],
                [2, 1, 0, 0],
            ],
            dtype="float32",
        )
        self.y_pred_batched = ops.array(
            [
                [0.1, 0.2, 0.9, 0.3],
                [0.1, 0.8, 0.9, 0.7],
                [0.4, 0.3, 0.2, 0.1],
                [0.9, 0.7, 0.1, 0.2],
            ],
            dtype="float32",
        )

        expected_dcg = [
            _compute_dcg([1], [1]),
            _compute_dcg([3, 2, 1], [1, 3, 4]),
            0.0,
            _compute_dcg([2, 1], [1, 2]),
        ]
        expected_idcg = [
            _compute_dcg([1], [1]),
            _compute_dcg([3, 2, 1], [1, 2, 3]),
            0.0,
            _compute_dcg([2, 1], [1, 2]),
        ]
        expected_ndcg = [
            a / b if b != 0.0 else 0.0
            for a, b in zip(expected_dcg, expected_idcg)
        ]
        self.expected_output_batched = sum(expected_ndcg) / 4.0

    def test_invalid_k_init(self):
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            nDCG(k=0)
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            nDCG(k=-5)
        with self.assertRaisesRegex(
            ValueError, "`k` should be a positive integer"
        ):
            nDCG(k=3.5)

    def test_unbatched_perfect_rank_binary(self):
        ndcg_metric = nDCG()
        ndcg_metric.update_state(
            self.y_true_unbatched, self.y_pred_unbatched_perfect
        )
        result = ndcg_metric.result()
        self.assertAllClose(result, 1.0)

    def test_unbatched_good_rank_binary(self):
        ndcg_metric = nDCG()
        ndcg_metric.update_state(
            self.y_true_unbatched, self.y_pred_unbatched_good
        )
        result = ndcg_metric.result()
        ndcg = _compute_dcg([1], [2])
        idcg = _compute_dcg([1], [1])
        self.assertAllClose(result, ndcg / idcg)

    def test_unbatched_bad_rank_binary(self):
        ndcg_metric = nDCG()
        ndcg_metric.update_state(
            self.y_true_unbatched, self.y_pred_unbatched_bad
        )
        result = ndcg_metric.result()
        ndcg = _compute_dcg([1], [3])
        idcg = _compute_dcg([1], [1])
        self.assertAllClose(result, ndcg / idcg)

    def test_unbatched_no_relevant(self):
        ndcg_metric = nDCG()
        ndcg_metric.update_state(
            self.y_true_unbatched_none,
            self.y_pred_unbatched_perfect,
        )
        result = ndcg_metric.result()
        self.assertAllClose(result, 0.0)

    def test_unbatched_perfect_rank_graded(self):
        ndcg_metric = nDCG()
        ndcg_metric.update_state(
            self.y_true_unbatched_graded,
            self.y_pred_unbatched_graded_perfect,
        )
        result = ndcg_metric.result()

        dcg = _compute_dcg([3, 2, 1], [1, 3, 4])
        idcg = _compute_dcg([3, 2, 1], [1, 2, 3])
        self.assertAllClose(result, dcg / idcg, rtol=1e-5)

    def test_unbatched_mixed_rank_graded(self):
        ndcg_metric = nDCG()
        ndcg_metric.update_state(
            self.y_true_unbatched_graded, self.y_pred_unbatched_graded_mixed
        )
        result = ndcg_metric.result()
        dcg = _compute_dcg([1, 2, 3], [1, 2, 3])
        idcg = _compute_dcg([3, 2, 1], [1, 2, 3])
        self.assertAllClose(result, dcg / idcg, rtol=1e-5)

    def test_batched_input(self):
        ndcg_metric = nDCG()
        ndcg_metric.update_state(self.y_true_batched, self.y_pred_batched)
        result = ndcg_metric.result()
        self.assertAllClose(result, self.expected_output_batched, rtol=1e-5)

    @parameterized.named_parameters(
        ("1", 1, 0.75),
        ("2", 2, 0.69679),
        ("3", 3, 0.72624),
        ("4", 4, 0.73770),
    )
    def test_k(self, k, exp_value_val):
        ndcg_metric = nDCG(k=k)
        ndcg_metric.update_state(self.y_true_batched, self.y_pred_batched)
        result = ndcg_metric.result()
        self.assertAllClose(result, exp_value_val, rtol=1e-5)

    def test_statefulness(self):
        ndcg_metric = nDCG()
        # Batch 1
        ndcg_metric.update_state(
            self.y_true_batched[:2], self.y_pred_batched[:2]
        )
        result = ndcg_metric.result()
        dcg = [_compute_dcg([1], [1]), _compute_dcg([3, 2, 1], [1, 3, 4])]
        idcg = [_compute_dcg([1], [1]), _compute_dcg([3, 2, 1], [1, 2, 3])]
        ndcg = sum([a / b if b != 0.0 else 0.0 for a, b in zip(dcg, idcg)]) / 2
        self.assertAllClose(result, ndcg, rtol=1e-5)

        # Batch 2
        ndcg_metric.update_state(
            self.y_true_batched[2:], self.y_pred_batched[2:]
        )
        result = ndcg_metric.result()
        self.assertAllClose(result, self.expected_output_batched, rtol=1e-5)

        # Reset state
        ndcg_metric.reset_state()
        result = ndcg_metric.result()
        self.assertAllClose(result, 0.0)

    @parameterized.named_parameters(
        ("0.5", 0.5, 0.7377),
        ("0", 0.0, 0.0),
        ("2", 2.0, 0.7377),
    )
    def test_scalar_sample_weight(self, sample_weight, expected_output):
        ndcg_metric = nDCG()
        ndcg_metric.update_state(
            self.y_true_batched,
            self.y_pred_batched,
            sample_weight=sample_weight,
        )
        result = ndcg_metric.result()
        self.assertAllClose(result, expected_output, rtol=1e-5)

    def test_1d_sample_weight(self):
        ndcg_metric = nDCG()
        sample_weight = ops.array([1.0, 0.5, 2.0, 1.5], dtype="float32")
        ndcg_metric.update_state(
            self.y_true_batched,
            self.y_pred_batched,
            sample_weight=sample_weight,
        )
        result = ndcg_metric.result()
        self.assertAllClose(result, 0.74385, rtol=1e-5)

    @parameterized.named_parameters(
        (
            "mask_relevant_item",
            ops.array([[0, 1, 0]], dtype="float32"),
            ops.array([[0.5, 0.8, 0.2]], dtype="float32"),
            ops.array([[1.0, 0.0, 1.0]], dtype="float32"),
            0.0,
        ),
        (
            "mask_highest_ranked_item",
            ops.array([[0, 1, 0]], dtype="float32"),
            ops.array([[0.5, 0.8, 0.2]], dtype="float32"),
            ops.array([[1.0, 0.0, 1.0]], dtype="float32"),
            0.0,
        ),
        (
            "mask_lower_ranked_relevant",
            ops.array([[1, 0, 1]], dtype="float32"),
            ops.array([[0.8, 0.2, 0.6]], dtype="float32"),
            ops.array([[1.0, 1.0, 0.0]], dtype="float32"),
            1.0,
        ),
        (
            "mask_irrelevant_item",
            ops.array([[0, 1, 0]], dtype="float32"),
            ops.array([[0.5, 0.8, 0.2]], dtype="float32"),
            ops.array([[0.0, 1.0, 1.0]], dtype="float32"),
            1.0,
        ),
        (
            "varied_masks",
            ops.array(
                [
                    [0, 1, 0, 0],
                    [1, 0, 0, 1],
                ],
                dtype="float32",
            ),
            ops.array(
                [
                    [0.8, 0.7, 0.1, 0.2],
                    [0.9, 0.1, 0.2, 0.3],
                ],
                dtype="float32",
            ),
            ops.array(
                [
                    [0.5, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                ],
                dtype="float32",
            ),
            0.815465,
        ),
    )
    def test_2d_sample_weight_masking(
        self, y_true, y_pred, sample_weight, expected_output_val
    ):
        ndcg_metric = nDCG()
        ndcg_metric.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = ndcg_metric.result()
        self.assertAllClose(result, expected_output_val, rtol=1e-5)

    def test_serialization(self):
        metric = nDCG()
        restored = deserialize(serialize(metric))
        self.assertDictEqual(metric.get_config(), restored.get_config())

    def test_alternative_gain_rank_discount_fns(self):
        def linear_gain_fn(label):
            return ops.cast(label, dtype="float32")

        def inverse_discount_fn(rank):
            return ops.divide(1.0, rank)

        ndcg_metric = nDCG(
            gain_fn=linear_gain_fn, rank_discount_fn=inverse_discount_fn
        )
        ndcg_metric.update_state(self.y_true_batched, self.y_pred_batched)
        result = ndcg_metric.result()

        dcg = [1 / 1, 3 / 1 + 2 / 3 + 1 / 4, 0, 2 / 1 + 1 / 2]
        idcg = [1 / 1, 3 / 1 + 2 / 2 + 1 / 3, 0.0, 2 / 1 + 1 / 2]
        ndcg = sum([a / b if b != 0.0 else 0.0 for a, b in zip(dcg, idcg)]) / 4
        self.assertAllClose(result, ndcg, rtol=1e-5)

    def test_model_evaluate(self):
        inputs = keras.Input(shape=(20,), dtype="float32")
        outputs = keras.layers.Dense(5)(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            loss=keras.losses.MeanSquaredError(),
            metrics=[nDCG()],
            optimizer="adam",
        )
        model.evaluate(
            x=keras.random.normal((2, 20)),
            y=keras.random.randint((2, 5), minval=0, maxval=4),
        )
