import numpy as np
import tensorflow as tf


class DataLoader:
    def __init__(
        self,
        file_pattern,
        batch_size,
        dense_features,
        large_emb_features,
        small_emb_features,
        label,
        training=False,
    ):
        # Passed attributes.
        self.file_pattern = file_pattern
        self.batch_size = batch_size
        self.dense_features = dense_features
        self.large_emb_features = large_emb_features
        self.small_emb_features = small_emb_features
        self.label = label
        self.training = training

        # Derived attributes.
        self._return_dummy_dataset = file_pattern is None

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

        # Sparse features
        large_emb_inputs = {}
        for large_emb_feature in self.large_emb_features:
            name = large_emb_feature["name"]
            new_name = large_emb_feature.get("new_name", name)
            vocabulary_size = large_emb_feature["vocabulary_size"]
            multi_hot_size = large_emb_feature["multi_hot_size"]

            large_emb_inputs[new_name] = np.random.randint(
                low=0,
                high=vocabulary_size,
                size=(self.batch_size, multi_hot_size),
                dtype=np.int64,
            )

        data["large_emb_inputs"] = large_emb_inputs

        # Dense lookup features
        small_emb_inputs = {}
        for small_emb_feature in self.small_emb_features:
            name = small_emb_feature["name"]
            new_name = small_emb_feature.get("new_name", name)
            vocabulary_size = small_emb_feature["vocabulary_size"]
            multi_hot_size = small_emb_feature["multi_hot_size"]

            small_emb_inputs[new_name] = np.random.randint(
                low=0,
                high=vocabulary_size,
                size=(self.batch_size, multi_hot_size),
                dtype=np.int64,
            )

        if small_emb_inputs:
            data["small_emb_inputs"] = small_emb_inputs

        return data

    def _create_dummy_dataset(self):
        """Creates a TF dummy dataset (randomly initialised)."""
        dummy_data = self._get_dummy_batch()

        # Separate labels from features to create a `(features, labels)` tuple.
        labels = dummy_data.pop("clicked")
        features = dummy_data

        dataset = tf.data.Dataset.from_tensors((features, labels)).repeat(512)
        return dataset

    def _get_feature_spec(self):
        feature_spec = {
            self.label: tf.io.FixedLenFeature(
                [self.batch_size],
                dtype=tf.int64,
            )
        }

        for dense_feat in self.dense_features:
            feature_spec[dense_feat] = tf.io.FixedLenFeature(
                [self.batch_size],
                dtype=tf.float32,
            )

        for emb_feat in self.large_emb_features + self.small_emb_features:
            name = emb_feat["name"]
            feature_spec[name] = tf.io.FixedLenFeature(
                [self.batch_size],
                dtype=tf.string,
            )

        return feature_spec

    def _preprocess(self, example):
        # Read example.
        feature_spec = self.get_feature_spec()
        example = tf.io.parse_single_example(example, feature_spec)

        # Dense features
        dense_input = tf.stack(
            [
                tf.reshape(example[dense_feature], [self.batch_size, 1])
                for dense_feature in self.dense_features
            ],
            axis=-1,
        )

        def _get_emb_inputs(emb_features):
            emb_inputs = {}
            for emb_feature in emb_features:
                name = emb_feature["name"]
                new_name = emb_feature.get("new_name", name)
                multi_hot_size = emb_feature["multi_hot_size"]

                raw_values = tf.io.decode_raw(example[name], tf.int64)
                raw_values = tf.reshape(
                    raw_values, [self.batch_size, multi_hot_size]
                )
                emb_inputs[new_name] = raw_values
            return emb_inputs

        # Sparse features
        large_emb_inputs = _get_emb_inputs(self.large_emb_features)
        small_emb_inputs = _get_emb_inputs(self.small_emb_features)

        # Labels
        labels = tf.reshape(example[self.label], [self.batch_size])

        x = {
            "dense_input": dense_input,
            "large_emb_inputs": large_emb_inputs,
        }
        if small_emb_inputs:
            x["small_emb_inputs"] = small_emb_inputs

        return (x, labels)

    def create_dataset(self, process_id=0, num_processes=1, shuffle_buffer=256):
        if self._return_dummy_dataset:
            return self._create_dummy_dataset()

        dataset = tf.data.Dataset.list_files(self.file_pattern, shuffle=False)

        # Shard the dataset across hosts/workers.
        # TODO: Do we need to do this if we are distributing the dataset
        # manually using distribution.distribute_dataset(...)?
        if num_processes > 1:
            dataset = dataset.shard(num_processes, process_id)

        dataset = tf.data.TFRecordDataset(
            dataset,
            buffer_size=None,
            num_parallel_reads=tf.data.AUTOTUNE,
        )

        # Process example.
        dataset = dataset.map(
            lambda x: self._preprocess(x),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Shuffle dataset if in training mode.
        if self.training and shuffle_buffer and shuffle_buffer > 0:
            dataset = dataset.shuffle(shuffle_buffer)

        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
