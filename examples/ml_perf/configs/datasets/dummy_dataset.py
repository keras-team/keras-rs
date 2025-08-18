from keras.utils import Config

# === Dataset ===
dataset_config = Config()
dataset_config.file_pattern = None
# Features
dataset_config.label = "clicked"
dataset_config.dense = [f"int-feature-{i}" for i in range(13)]
dataset_config.sparse = [
    {
        "name": "categorical-feature-14",
        "vocabulary_size": 40000000,
        "multi_hot_size": 3,
    },
    {
        "name": "categorical-feature-15",
        "vocabulary_size": 39060,
        "multi_hot_size": 2,
    },
    {
        "name": "categorical-feature-16",
        "vocabulary_size": 17295,
        "multi_hot_size": 1,
    },
    {
        "name": "categorical-feature-17",
        "vocabulary_size": 7424,
        "multi_hot_size": 2,
    },
    {
        "name": "categorical-feature-18",
        "vocabulary_size": 20265,
        "multi_hot_size": 6,
    },
    {
        "name": "categorical-feature-19",
        "vocabulary_size": 3,
        "multi_hot_size": 1,
    },
    {
        "name": "categorical-feature-20",
        "vocabulary_size": 7122,
        "multi_hot_size": 1,
    },
    {
        "name": "categorical-feature-21",
        "vocabulary_size": 1543,
        "multi_hot_size": 1,
    },
    {
        "name": "categorical-feature-22",
        "vocabulary_size": 63,
        "multi_hot_size": 1,
    },
    {
        "name": "categorical-feature-23",
        "vocabulary_size": 40000000,
        "multi_hot_size": 7,
    },
    {
        "name": "categorical-feature-24",
        "vocabulary_size": 3067956,
        "multi_hot_size": 3,
    },
    {
        "name": "categorical-feature-25",
        "vocabulary_size": 405282,
        "multi_hot_size": 8,
    },
    {
        "name": "categorical-feature-26",
        "vocabulary_size": 10,
        "multi_hot_size": 1,
    },
    {
        "name": "categorical-feature-27",
        "vocabulary_size": 2209,
        "multi_hot_size": 6,
    },
    {
        "name": "categorical-feature-28",
        "vocabulary_size": 11938,
        "multi_hot_size": 9,
    },
    {
        "name": "categorical-feature-29",
        "vocabulary_size": 155,
        "multi_hot_size": 5,
    },
    {
        "name": "categorical-feature-30",
        "vocabulary_size": 4,
        "multi_hot_size": 1,
    },
    {
        "name": "categorical-feature-31",
        "vocabulary_size": 976,
        "multi_hot_size": 1,
    },
    {
        "name": "categorical-feature-32",
        "vocabulary_size": 14,
        "multi_hot_size": 1,
    },
    {
        "name": "categorical-feature-33",
        "vocabulary_size": 40000000,
        "multi_hot_size": 12,
    },
    {
        "name": "categorical-feature-34",
        "vocabulary_size": 40000000,
        "multi_hot_size": 100,
    },
    {
        "name": "categorical-feature-35",
        "vocabulary_size": 40000000,
        "multi_hot_size": 27,
    },
    {
        "name": "categorical-feature-36",
        "vocabulary_size": 590152,
        "multi_hot_size": 10,
    },
    {
        "name": "categorical-feature-37",
        "vocabulary_size": 12973,
        "multi_hot_size": 3,
    },
    {
        "name": "categorical-feature-38",
        "vocabulary_size": 108,
        "multi_hot_size": 1,
    },
    {
        "name": "categorical-feature-39",
        "vocabulary_size": 36,
        "multi_hot_size": 1,
    },
]
