import keras
import tensorflow as tf
from absl.testing import parameterized
from keras import ops
from keras.layers import deserialize
from keras.layers import serialize

from keras_rs.src import testing
from keras_rs.src.layers.retrieval import remove_accidental_hits
from keras_rs.src.utils import tpu_test_utils


class RemoveAccidentalHitsTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        if keras.backend.backend() == "tensorflow":
            tf.debugging.disable_traceback_filtering()

        self._strategy = tpu_test_utils.get_tpu_strategy(self)

    def create_inputs(self, logits_rank=2, candidate_ids_rank=1):
        shape_3d = (15, 20, 10)
        shape = shape_3d[-logits_rank:]
        candidate_ids_shape = shape_3d[-candidate_ids_rank:]
        num_candidates = shape[-1]

        rng = keras.random.SeedGenerator(42)
        logits = keras.random.uniform(shape, seed=rng)
        labels = keras.ops.one_hot(
            keras.random.randint(
                shape[:-1], minval=0, maxval=num_candidates, seed=rng
            ),
            num_candidates,
        )
        candidate_ids = keras.random.randint(
            candidate_ids_shape, minval=0, maxval=num_candidates, seed=rng
        )

        return logits, labels, candidate_ids

    @parameterized.named_parameters(
        [
            {
                "testcase_name": "logits_rank_1_candidate_ids_rank_1",
                "logits_rank": 1,
                "candidate_ids_rank": 1,
            },
            {
                "testcase_name": "logits_rank_2_candidate_ids_rank_1",
                "logits_rank": 2,
                "candidate_ids_rank": 1,
            },
            {
                "testcase_name": "logits_rank_2_candidate_ids_rank_2",
                "logits_rank": 2,
                "candidate_ids_rank": 2,
            },
            {
                "testcase_name": "logits_rank_3_candidate_ids_rank_1",
                "logits_rank": 3,
                "candidate_ids_rank": 1,
            },
            {
                "testcase_name": "logits_rank_3_candidate_ids_rank_2",
                "logits_rank": 3,
                "candidate_ids_rank": 2,
            },
            {
                "testcase_name": "logits_rank_3_candidate_ids_rank_3",
                "logits_rank": 3,
                "candidate_ids_rank": 3,
            },
        ]
    )
    def test_call(self, logits_rank, candidate_ids_rank):
        logits, labels, candidate_ids = self.create_inputs(
            logits_rank=logits_rank, candidate_ids_rank=candidate_ids_rank
        )

        out_logits = remove_accidental_hits.RemoveAccidentalHits()(
            logits, labels, candidate_ids
        )

        # Logits of labels are unchanged.
        self.assertAllClose(
            ops.sum(ops.multiply(out_logits, labels), axis=-1),
            ops.sum(ops.multiply(logits, labels), axis=-1),
        )

        # Instead of having nested loops, which we can't do becasue they depend
        # on the rank, we unroll the index combinations.
        shape = ops.shape(logits)
        if logits_rank == 1:
            indices = [
                (),
            ]
        elif logits_rank == 2:
            indices = [(i,) for i in range(shape[0])]
        elif logits_rank == 3:
            indices = [(i, j) for i in range(shape[0]) for j in range(shape[1])]

        for index_tuple in indices:
            sub_labels = labels
            sub_logits = logits
            sub_out_logits = out_logits
            sub_candidate_ids = candidate_ids
            # This loop applies multiple indices to go deep several dimensions.
            for i in index_tuple:
                sub_labels = sub_labels[i]
                sub_logits = sub_logits[i]
                sub_out_logits = sub_out_logits[i]
                if len(ops.shape(sub_candidate_ids)) > 1:
                    sub_candidate_ids = sub_candidate_ids[i]

            row_positive_idx = ops.argmax(sub_labels)
            positive_candidate_id = sub_candidate_ids[row_positive_idx]

            for col_idx in range(sub_out_logits.shape[0]):
                same_candidate_as_positive = ops.equal(
                    positive_candidate_id, sub_candidate_ids[col_idx]
                )
                is_positive = ops.equal(col_idx, row_positive_idx)

                if ops.convert_to_numpy(
                    same_candidate_as_positive
                ) and not ops.convert_to_numpy(is_positive):
                    # We zeroed the logits.
                    self.assertAllClose(
                        sub_out_logits[col_idx],
                        ops.add(
                            sub_logits[col_idx],
                            remove_accidental_hits.SMALLEST_FLOAT,
                        ),
                    )
                else:
                    # We left the logits unchanged.
                    self.assertAllClose(
                        sub_out_logits[col_idx],
                        sub_logits[col_idx],
                    )

    def test_mismatched_labels_logits_shapes(self):
        layer = remove_accidental_hits.RemoveAccidentalHits()

        with self.assertRaisesRegex(
            ValueError, "`labels` and `logits` should have the same shape"
        ):
            layer(ops.zeros((10, 20)), ops.zeros((10, 30)), ops.zeros((20,)))

    def test_mismatched_labels_candidates_shapes(self):
        layer = remove_accidental_hits.RemoveAccidentalHits()

        with self.assertRaisesRegex(
            ValueError,
            "`candidate_ids` should have the same shape as .* `labels`",
        ):
            layer(ops.zeros((10, 20)), ops.zeros((10, 20)), ops.zeros((30,)))

    def test_predict(self):
        # Note: for predict, we test with probabilities that have a batch dim.
        logits, labels, candidate_ids = self.create_inputs(candidate_ids_rank=2)

        with self._strategy.scope():
            layer = remove_accidental_hits.RemoveAccidentalHits()
            in_logits = keras.layers.Input(logits.shape[1:])
            in_labels = keras.layers.Input(labels.shape[1:])
            in_candidate_ids = keras.layers.Input(labels.shape[1:])
            out_logits = layer(in_logits, in_labels, in_candidate_ids)
            model = keras.Model(
                [in_logits, in_labels, in_candidate_ids], out_logits
            )
            model.compile(optimizer="adam", loss="mse")

        model.predict([logits, labels, candidate_ids], batch_size=8)

    def test_serialization(self):
        layer = remove_accidental_hits.RemoveAccidentalHits()
        restored = deserialize(serialize(layer))
        self.assertDictEqual(layer.get_config(), restored.get_config())

    def test_model_saving(self):
        logits, labels, candidate_ids = self.create_inputs()

        layer = remove_accidental_hits.RemoveAccidentalHits()
        in_logits = keras.layers.Input(logits.shape[1:])
        in_labels = keras.layers.Input(labels.shape[1:])
        in_candidate_ids = keras.layers.Input(batch_shape=candidate_ids.shape)
        out_logits = layer(in_logits, in_labels, in_candidate_ids)
        model = keras.Model(
            [in_logits, in_labels, in_candidate_ids], out_logits
        )

        self.run_model_saving_test(
            model=model, input_data=[logits, labels, candidate_ids]
        )
