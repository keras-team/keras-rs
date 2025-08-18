import numpy as np
import tensorflow as tf


def _get_dummy_batch(batch_size, large_emb_features, small_emb_features):
    """Returns a dummy batch of data in the final desired structure."""

    # Labels
    data = {
        "clicked": np.random.randint(0, 2, size=(batch_size,), dtype=np.int64)
    }

    # Dense features
    dense_input_list = [
        np.random.uniform(0.0, 0.9, size=(batch_size, 1)).astype(np.float32)
        for _ in range(13)
    ]
    data["dense_input"] = np.concatenate(dense_input_list, axis=-1)

    # Sparse features
    large_emb_inputs = {}
    for large_emb_feature in large_emb_features:
        vocabulary_size = large_emb_feature["vocabulary_size"]
        multi_hot_size = large_emb_feature["multi_hot_size"]
        idx = large_emb_feature["name"].split("-")[-1]

        large_emb_inputs[f"cat_{idx}_id"] = np.random.randint(
            low=0,
            high=vocabulary_size,
            size=(batch_size, multi_hot_size),
            dtype=np.int64,
        )

    data["large_emb_inputs"] = large_emb_inputs

    # Dense lookup features
    small_emb_inputs = {}
    for small_emb_feature in small_emb_features:
        vocabulary_size = small_emb_feature["vocabulary_size"]
        multi_hot_size = small_emb_feature["multi_hot_size"]
        idx = small_emb_feature["name"].split("-")[-1]

        # TODO: We don't need this custom renaming. Remove later, when we
        # shift from dummy data to actual data.
        small_emb_inputs[f"cat_{idx}_id"] = np.random.randint(
            low=0,
            high=vocabulary_size,
            size=(batch_size, multi_hot_size),
            dtype=np.int64,
        )

    if small_emb_inputs:
        data["small_emb_inputs"] = small_emb_inputs

    return data


def create_dummy_dataset(batch_size, large_emb_features, small_emb_features):
    """Creates a TF dataset from cached dummy data of the final batch size."""
    dummy_data = _get_dummy_batch(
        batch_size, large_emb_features, small_emb_features
    )

    # Separate labels from features to create a `(features, labels)` tuple.
    labels = dummy_data.pop("clicked")
    features = dummy_data

    dataset = tf.data.Dataset.from_tensors((features, labels)).repeat(512)
    return dataset
