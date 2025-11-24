import math

import keras
import tensorflow as tf
from absl.testing import parameterized
from keras import ops
from keras.metrics import deserialize
from keras.metrics import serialize

from keras_rs.src import testing
from keras_rs.src.metrics.ndcg import NDCG
from keras_rs.src.utils import tpu_test_utils


def _compute_dcg(labels, ranks):
    val = 0.0
    for label, rank in zip(labels, ranks):
        val += (math.pow(2, label) - 1) / math.log2(rank + 1)
    return val


class NDCGTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        if keras.backend.backend() == "tensorflow":
            tf.debugging.disable_traceback_filtering()
        self._strategy = tpu_test_utils.get_tpu_strategy(self)

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
        with self._strategy.scope():
            with self.assertRaisesRegex(
                ValueError, "`k` should be a positive integer"
            ):
                NDCG(k=0)
            with self.assertRaisesRegex(
                ValueError, "`k` should be a positive integer"
            ):
                NDCG(k=-5)
            with self.assertRaisesRegex(
                ValueError, "`k` should be a positive integer"
            ):
                NDCG(k=3.5)

    @parameterized.named_parameters(
        (
            "binary_perfect_rank",
            [0, 0, 1, 0],
            [0.1, 0.2, 0.9, 0.3],
            None,
            1.0,
        ),
        (
            "binary_good_rank",
            [0, 0, 1, 0],
            [0.8, 0.1, 0.7, 0.2],
            None,
            _compute_dcg([1], [2]) / _compute_dcg([1], [1]),
        ),
        (
            "binary_bad_rank",
            [0, 0, 1, 0],
            [0.4, 0.3, 0.2, 0.1],
            None,
            _compute_dcg([1], [3]) / _compute_dcg([1], [1]),
        ),
        (
            "irrelevant",
            [0, 0, 0, 0],
            [0.1, 0.2, 0.9, 0.3],
            None,
            0.0,
        ),
        (
            "graded_good_rank",
            [1, 0, 3, 2],
            [0.1, 0.8, 0.9, 0.7],
            None,
            _compute_dcg([3, 2, 1], [1, 3, 4])
            / _compute_dcg([3, 2, 1], [1, 2, 3]),
        ),
        (
            "graded_mixed_rank",
            [1, 0, 3, 2],
            [0.9, 0.1, 0.7, 0.8],
            None,
            _compute_dcg([1, 2, 3], [1, 2, 3])
            / _compute_dcg([3, 2, 1], [1, 2, 3]),
        ),
        (
            "sample_weight_0",
            [0.0, 1.0, 2.0],
            [0.5, 0.8, 0.2],
            [0.0, 0.0, 0.0],
            0.0,
        ),
        (
            "sample_weight_scalar",
            [1, 0, 3, 2],
            [0.1, 0.8, 0.9, 0.7],
            5.0,
            _compute_dcg([3, 2, 1], [1, 3, 4])
            / _compute_dcg([3, 2, 1], [1, 2, 3]),
        ),
        (
            "sample_weight_1d",
            [1, 0, 3, 2],
            [0.1, 0.8, 0.9, 0.7],
            [2.0, 1.0, 3.0, 0.0],
            0.988237,
        ),
    )
    def test_unbatched_inputs(
        self, y_true, y_pred, sample_weight, expected_output
    ):
        with self._strategy.scope():
            ndcg_metric = NDCG()
        tpu_test_utils.run_with_strategy(
            self._strategy,
            ndcg_metric.update_state,
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
        result = ndcg_metric.result()
        self.assertAllClose(result, expected_output)

    def test_batched_input(self):
        with self._strategy.scope():
            ndcg_metric = NDCG()
        tpu_test_utils.run_with_strategy(
            self._strategy,
            ndcg_metric.update_state,
            self.y_true_batched,
            self.y_pred_batched,
        )
        result = ndcg_metric.result()
        self.assertAllClose(result, self.expected_output_batched)

    @parameterized.named_parameters(
        ("scalar_0.5", 0.5, 0.73770034),
        ("scalar_0", 0, 0),
        ("1d", [1.0, 0.5, 2.0, 1.0], 0.74262),
    )
    def test_batched_inputs_sample_weight(self, sample_weight, expected_output):
        with self._strategy.scope():
            ndcg_metric = NDCG()
        tpu_test_utils.run_with_strategy(
            self._strategy,
            ndcg_metric.update_state,
            self.y_true_batched,
            self.y_pred_batched,
            sample_weight=sample_weight,
        )
        result = ndcg_metric.result()
        self.assertAllClose(result, expected_output)

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
            "general_case",
            ops.array(
                [
                    [0, 1, 0, 0, 2, 3],
                    [1, 0, 0, 2, 0, 2],
                ],
                dtype="float32",
            ),
            ops.array(
                [
                    [0.8, 0.7, 0.1, 0.2, 0.9, 0.5],
                    [0.9, 0.1, 0.2, 0.3, 0.2, 0.3],
                ],
                dtype="float32",
            ),
            ops.array(
                [
                    [0.5, 4.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 3.0, 1.0, 2.0, 0.0],
                ],
                dtype="float32",
            ),
            0.903588,
        ),
    )
    def test_2d_sample_weight(
        self, y_true, y_pred, sample_weight, expected_output
    ):
        with self._strategy.scope():
            ndcg_metric = NDCG()
        tpu_test_utils.run_with_strategy(
            self._strategy,
            ndcg_metric.update_state,
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
        result = ndcg_metric.result()
        self.assertAllClose(result, expected_output)

    @parameterized.named_parameters(
        (
            "mask_relevant_items",
            {"labels": [[0.0, 1.0, 0.0]], "mask": [[True, False, True]]},
            [[0.5, 0.8, 0.2]],
            None,
            0.0,
        ),
        (
            "mask_first_relevant_item",
            {"labels": [[1, 0, 1]], "mask": [[False, True, True]]},
            [[0.8, 0.2, 0.6]],
            None,
            1.0,
        ),
        (
            "mask_irrelevant_item",
            {"labels": [[0, 1, 0]], "mask": [[False, True, True]]},
            [[0.5, 0.8, 0.2]],
            None,
            1.0,
        ),
        (
            "general_case",
            {
                "labels": ops.array(
                    [
                        [0, 1, 0, 0, 2, 3],
                        [1, 0, 0, 2, 0, 2],
                    ],
                    dtype="float32",
                ),
                "mask": ops.array(
                    [
                        [True, True, True, False, True, False],
                        [True, False, True, True, True, False],
                    ]
                ),
            },
            ops.array(
                [
                    [0.8, 0.7, 0.1, 0.2, 0.9, 0.5],
                    [0.9, 0.1, 0.2, 0.3, 0.2, 0.3],
                ],
                dtype="float32",
            ),
            ops.array(
                [
                    [0.5, 4.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 3.0, 1.0, 2.0, 0.0],
                ],
                dtype="float32",
            ),
            0.903588,
        ),
    )
    def test_masking(self, y_true, y_pred, sample_weight, expected_output):
        with self._strategy.scope():
            ndcg_metric = NDCG()
        tpu_test_utils.run_with_strategy(
            self._strategy,
            ndcg_metric.update_state,
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
        result = ndcg_metric.result()
        self.assertAllClose(result, expected_output)

    @parameterized.named_parameters(
        ("1", 1, 0.75),
        ("2", 2, 0.696789),
        ("3", 3, 0.72623),
        ("4", 4, 0.7377),
    )
    def test_k(self, k, exp_value):
        with self._strategy.scope():
            ndcg_metric = NDCG(k=k)
        tpu_test_utils.run_with_strategy(
            self._strategy,
            ndcg_metric.update_state,
            self.y_true_batched,
            self.y_pred_batched,
        )
        result = ndcg_metric.result()
        self.assertAllClose(result, exp_value, rtol=1e-5)

    def test_statefulness(self):
        with self._strategy.scope():
            ndcg_metric = NDCG()
        # Batch 1
        tpu_test_utils.run_with_strategy(
            self._strategy,
            ndcg_metric.update_state,
            self.y_true_batched[:2],
            self.y_pred_batched[:2],
        )
        result = ndcg_metric.result()
        dcg = [_compute_dcg([1], [1]), _compute_dcg([3, 2, 1], [1, 3, 4])]
        idcg = [_compute_dcg([1], [1]), _compute_dcg([3, 2, 1], [1, 2, 3])]
        ndcg = sum([a / b if b != 0.0 else 0.0 for a, b in zip(dcg, idcg)]) / 2
        self.assertAllClose(
            result,
            ndcg,
        )

        # Batch 2
        tpu_test_utils.run_with_strategy(
            self._strategy,
            ndcg_metric.update_state,
            self.y_true_batched[2:],
            self.y_pred_batched[2:],
        )
        result = ndcg_metric.result()
        self.assertAllClose(result, self.expected_output_batched)

        # Reset state
        ndcg_metric.reset_state()
        result = ndcg_metric.result()
        self.assertAllClose(result, 0.0)

    def test_serialization(self):
        with self._strategy.scope():
            metric = NDCG()
        restored = deserialize(serialize(metric))
        self.assertDictEqual(metric.get_config(), restored.get_config())

    def test_alternative_gain_rank_discount_fns(self):
        def linear_gain_fn(label):
            return ops.cast(label, dtype="float32")

        def inverse_discount_fn(rank):
            return ops.divide(1.0, rank)

        with self._strategy.scope():
            ndcg_metric = NDCG(
                gain_fn=linear_gain_fn, rank_discount_fn=inverse_discount_fn
            )
        tpu_test_utils.run_with_strategy(
            self._strategy,
            ndcg_metric.update_state,
            self.y_true_batched,
            self.y_pred_batched,
        )
        result = ndcg_metric.result()

        dcg = [1 / 1, 3 / 1 + 2 / 3 + 1 / 4, 0, 2 / 1 + 1 / 2]
        idcg = [1 / 1, 3 / 1 + 2 / 2 + 1 / 3, 0.0, 2 / 1 + 1 / 2]
        ndcg = sum([a / b if b != 0.0 else 0.0 for a, b in zip(dcg, idcg)]) / 4
        self.assertAllClose(result, ndcg, rtol=1e-5)

    def test_model_evaluate(self):
        with self._strategy.scope():
            inputs = keras.Input(shape=(20,), dtype="float32")
            outputs = keras.layers.Dense(5)(inputs)
            model = keras.Model(inputs=inputs, outputs=outputs)

            model.compile(
                loss=keras.losses.MeanSquaredError(),
                metrics=[NDCG()],
                optimizer="adam",
            )

        x_data = keras.random.normal((2, 20))
        y_data = keras.random.randint((2, 5), minval=0, maxval=4)

        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        dataset = dataset.batch(
            self._strategy.num_replicas_in_sync
            if isinstance(self._strategy, tf.distribute.Strategy)
            else 1
        )

        if isinstance(self._strategy, tf.distribute.TPUStrategy):
            dataset = self._strategy.experimental_distribute_dataset(dataset)

        model.evaluate(dataset, steps=2)
