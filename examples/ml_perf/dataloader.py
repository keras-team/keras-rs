import logging

import numpy as np
import tensorflow as tf

SEED = 1337

logger = logging.getLogger(__name__)


def get_iterator(dataset):
    def _convert_to_numpy(batch):
        x, y = batch

        numpy_x = {
            "dense_input": x["dense_input"].numpy(),
            "large_emb_inputs": {
                k: v.numpy() for k, v in x["large_emb_inputs"].items()
            },
        }
        if "small_emb_inputs" in x:
            numpy_x["small_emb_inputs"] = {
                k: v.numpy() for k, v in x["small_emb_inputs"].items()
            }
        numpy_y = y.numpy()
        return (numpy_x, numpy_y)

    return map(_convert_to_numpy, iter(dataset))


class DataLoader:
    def __init__(
        self,
        file_pattern,
        batch_size,
        file_batch_size,
        dense_features,
        large_emb_features,
        small_emb_features,
        label,
        num_steps,
        repeat=False,
        training=False,
    ):
        passed_args = locals()
        logger.debug("Initialising `DataLoader` with: %s", passed_args)

        # Passed attributes.
        self.file_pattern = file_pattern
        self.batch_size = batch_size
        self.file_batch_size = file_batch_size
        self.dense_features = dense_features
        self.large_emb_features = large_emb_features
        self.small_emb_features = small_emb_features
        self.label = label
        self.num_steps = num_steps
        self.repeat = repeat
        self.training = training

        # Derived attributes.
        self._return_dummy_dataset = file_pattern is None
        if self._return_dummy_dataset:
            logger.warning(
                "`file_pattern` is `None`. Will use the dummy dataset."
            )

    def _get_dummy_batch(self):
        """Returns a dummy batch of data in the final desired structure."""
        # Labels
        data = {
            "clicked": np.random.randint(
                0, 2, size=(self.batch_size,), dtype=np.int64
            )
        }

        # Dense features
        dense_input_list = [
            np.random.uniform(0.0, 0.9, size=(self.batch_size, 1)).astype(
                np.float32
            )
            for _ in range(13)
        ]
        data["dense_input"] = np.concatenate(dense_input_list, axis=-1)

        # Big embedding features
        large_emb_inputs = {}
        for large_emb_feature in self.large_emb_features:
            name = large_emb_feature["name"]
            new_name = large_emb_feature.get("new_name", name)
            vocabulary_size = large_emb_feature["vocabulary_size"]
            feature_list_length = large_emb_feature["feature_list_length"]

            large_emb_inputs[f"{new_name}_id"] = np.random.randint(
                low=0,
                high=vocabulary_size,
                size=(self.batch_size, feature_list_length),
                dtype=np.int64,
            )

        data["large_emb_inputs"] = large_emb_inputs

        # Small embedding features
        small_emb_inputs = {}
        for small_emb_feature in self.small_emb_features:
            name = small_emb_feature["name"]
            new_name = small_emb_feature.get("new_name", name)
            vocabulary_size = small_emb_feature["vocabulary_size"]
            feature_list_length = small_emb_feature["feature_list_length"]

            small_emb_inputs[f"{new_name}_id"] = np.random.randint(
                low=0,
                high=vocabulary_size,
                size=(self.batch_size, feature_list_length),
                dtype=np.int64,
            )

        if small_emb_inputs:
            data["small_emb_inputs"] = small_emb_inputs

        return data

    def _create_dummy_dataset(self):
        """Creates a TF dummy dataset (randomly initialised)."""
        logger.info("Creating dummy dataset...")
        dummy_data = self._get_dummy_batch()

        # Separate labels from features to create a `(features, labels)` tuple.
        labels = dummy_data.pop("clicked")
        features = dummy_data

        dataset = tf.data.Dataset.from_tensors((features, labels))
        dataset = dataset.repeat()
        return dataset

    def _get_feature_spec(self):
        feature_spec = {
            self.label: tf.io.FixedLenFeature(
                [self.file_batch_size],
                dtype=tf.int64,
            )
        }

        for dense_feat in self.dense_features:
            feature_spec[dense_feat] = tf.io.FixedLenFeature(
                [self.file_batch_size],
                dtype=tf.float32,
            )

        for emb_feat in self.large_emb_features + self.small_emb_features:
            name = emb_feat["name"]
            feature_spec[name] = tf.io.FixedLenFeature(
                [self.file_batch_size],
                dtype=tf.string,
            )

        return feature_spec

    def _preprocess(self, example):
        # Read example.
        feature_spec = self._get_feature_spec()
        example = tf.io.parse_single_example(example, feature_spec)

        # Dense features
        dense_input = tf.concat(
            [
                tf.reshape(example[dense_feature], [self.file_batch_size, 1])
                for dense_feature in self.dense_features
            ],
            axis=-1,
        )

        def _get_emb_inputs(emb_features):
            emb_inputs = {}
            for emb_feature in emb_features:
                name = emb_feature["name"]
                new_name = emb_feature.get("new_name", name)
                feature_list_length = emb_feature["feature_list_length"]

                raw_values = tf.io.decode_raw(example[name], tf.int64)
                raw_values = tf.reshape(
                    raw_values, [self.file_batch_size, feature_list_length]
                )
                emb_inputs[f"{new_name}_id"] = raw_values
            return emb_inputs

        # Embedding/lookup features
        large_emb_inputs = _get_emb_inputs(self.large_emb_features)
        small_emb_inputs = _get_emb_inputs(self.small_emb_features)

        # Labels
        labels = tf.reshape(example[self.label], [self.file_batch_size])

        x = {
            "dense_input": dense_input,
            "large_emb_inputs": large_emb_inputs,
        }
        if small_emb_inputs:
            x["small_emb_inputs"] = small_emb_inputs

        return (x, labels)

    def create_dataset(self, process_id=0, num_processes=1, shuffle_buffer=256):
        passed_args = locals()
        logger.debug("Called `create_dataset` with:%s", passed_args)

        if self._return_dummy_dataset:
            return self._create_dummy_dataset()

        logger.info("Loading the real dataset from files...")
        # Important to specify shuffle = False here to ensure all processes have
        # the same order.
        dataset = tf.data.Dataset.list_files(self.file_pattern, shuffle=False)
        # dataset = dataset.shard(num_shards=num_processes, index=process_id)
        logger.info("List of input files: %s", [f for f in dataset])

        dataset = tf.data.TFRecordDataset(
            dataset,
            buffer_size=None,
            num_parallel_reads=tf.data.AUTOTUNE,
        )

        # Process example.
        dataset = dataset.map(
            self._preprocess, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.unbatch()

        # Take only `num_steps * self.batch_size` examples.
        dataset = dataset.take(self.num_steps * self.batch_size)

        # Shuffle dataset if in training mode. Pass a seed so that all processes
        # have the same shuffle.
        if self.training and shuffle_buffer and shuffle_buffer > 0:
            dataset = dataset.shuffle(shuffle_buffer, seed=SEED)

        dataset = dataset.batch(
            self.batch_size,
            drop_remainder=True,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Repeat the dataset infinite number of times so that the generator
        # does not run out.
        if self.repeat:
            dataset = dataset.repeat()

        # dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
