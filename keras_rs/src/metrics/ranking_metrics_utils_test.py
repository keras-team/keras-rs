import keras
import numpy as np
from absl.testing import parameterized
from keras import ops

from keras_rs.src import testing
from keras_rs.src.metrics.ranking_metrics_utils import sort_by_scores


class SortByScoresTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()

        self.items = ops.array([[0, 1, 2, 3, 4]], dtype="int32")
        self.tied_scores = ops.array(
            [[0.5, 0.5, 0.5, 0.5, 0.5]], dtype="float32"
        )

    @parameterized.named_parameters(
        ("shuffle_ties", True), ("no_shuffle_ties", False)
    )
    def test_distinct_scores(self, shuffle_ties):
        scores = ops.array([[0.2, 0.9, 0.7, 0.1, 0.5]], dtype="float32")
        labels = ops.array([[1.0, 0.0, 2.0, 0.0, 3.0]], dtype="float32")
        sorted_items, sorted_labels = sort_by_scores(
            tensors_to_sort=[self.items, labels],
            scores=scores,
            shuffle_ties=shuffle_ties,
            seed=keras.random.SeedGenerator(42),
        )
        self.assertAllEqual(sorted_items, np.array([[1, 2, 4, 0, 3]]))
        self.assertAllClose(
            sorted_labels, np.array([[0.0, 2.0, 3.0, 1.0, 0.0]])
        )

    def test_no_shuffle_ties_keeps_original_order(self):
        for _ in range(10):
            (sorted_items,) = sort_by_scores(
                tensors_to_sort=[self.items],
                scores=self.tied_scores,
                shuffle_ties=False,
                seed=keras.random.SeedGenerator(42),
            )
            self.assertAllEqual(sorted_items, self.items)

    def test_no_shuffle_ties_with_mask_keeps_original_order(self):
        mask = ops.array([[True, False, True, True, False]])
        for _ in range(10):
            (sorted_items,) = sort_by_scores(
                tensors_to_sort=[self.items],
                scores=self.tied_scores,
                mask=mask,
                shuffle_ties=False,
                seed=keras.random.SeedGenerator(42),
            )
            self.assertAllEqual(sorted_items, np.array([[0, 2, 3, 1, 4]]))

    def test_no_shuffle_ties_masked_items_lose_ties(self):
        scores = ops.array([[0.2, 0.9, 0.2, 0.1, 0.5]], dtype="float32")
        mask = ops.array([[True, True, False, True, True]])
        (sorted_items,) = sort_by_scores(
            tensors_to_sort=[self.items],
            scores=scores,
            mask=mask,
            shuffle_ties=False,
        )
        self.assertAllEqual(sorted_items, np.array([[1, 4, 0, 2, 3]]))

    def test_no_shuffle_ties_with_k(self):
        mask = ops.array([[True, False, True, True, False]])
        (sorted_items,) = sort_by_scores(
            tensors_to_sort=[self.items],
            scores=self.tied_scores,
            mask=mask,
            k=2,
            shuffle_ties=False,
        )
        self.assertAllEqual(sorted_items, np.array([[0, 2]]))

    def test_shuffle_ties_masked_items_last(self):
        mask = ops.array([[True, True, True, False, False]])
        seed = keras.random.SeedGenerator(42)
        for _ in range(10):
            (sorted_items,) = sort_by_scores(
                tensors_to_sort=[self.items],
                scores=self.tied_scores,
                mask=mask,
                shuffle_ties=True,
                seed=seed,
            )
            self.assertAllEqual(
                ops.sort(sorted_items[:, :3], axis=1), np.array([[0, 1, 2]])
            )
            self.assertAllEqual(
                ops.sort(sorted_items[:, 3:], axis=1), np.array([[3, 4]])
            )

    def test_shuffle_ties_shuffles(self):
        seed = keras.random.SeedGenerator(42)
        outputs = set()
        for _ in range(20):
            (sorted_items,) = sort_by_scores(
                tensors_to_sort=[self.items],
                scores=self.tied_scores,
                shuffle_ties=True,
                seed=seed,
            )
            outputs.add(tuple(ops.convert_to_numpy(sorted_items)[0].tolist()))
        self.assertGreater(len(outputs), 1)
