from keras.utils import Config

config = Config()

# === Experiment metadata ===
config.experiment_name = "v6e_8"
config.model_dir = "./v6e_8"

# === Dataset ===
dataset_config = Config()
dataset_config.file_pattern = None
# Features
dataset_config.label = "clicked"
dataset_config.dense = [f"int-feature-{i}" for i in range(1, 14)]
dataset_config.lookup = [
    {
        "name": "categorical-feature-14",
        "vocabulary_size": 40000000,
        "feature_list_length": 3,
        "new_name": "cat_14",
    },
    {
        "name": "categorical-feature-15",
        "vocabulary_size": 39060,
        "feature_list_length": 2,
        "new_name": "cat_15",
    },
    {
        "name": "categorical-feature-16",
        "vocabulary_size": 17295,
        "feature_list_length": 1,
        "new_name": "cat_16",
    },
    {
        "name": "categorical-feature-17",
        "vocabulary_size": 7424,
        "feature_list_length": 2,
        "new_name": "cat_17",
    },
    {
        "name": "categorical-feature-18",
        "vocabulary_size": 20265,
        "feature_list_length": 6,
        "new_name": "cat_18",
    },
    {
        "name": "categorical-feature-19",
        "vocabulary_size": 3,
        "feature_list_length": 1,
        "new_name": "cat_19",
    },
    {
        "name": "categorical-feature-20",
        "vocabulary_size": 7122,
        "feature_list_length": 1,
        "new_name": "cat_20",
    },
    {
        "name": "categorical-feature-21",
        "vocabulary_size": 1543,
        "feature_list_length": 1,
        "new_name": "cat_21",
    },
    {
        "name": "categorical-feature-22",
        "vocabulary_size": 63,
        "feature_list_length": 1,
        "new_name": "cat_22",
    },
    {
        "name": "categorical-feature-23",
        "vocabulary_size": 40000000,
        "feature_list_length": 7,
        "new_name": "cat_23",
    },
    {
        "name": "categorical-feature-24",
        "vocabulary_size": 3067956,
        "feature_list_length": 3,
        "new_name": "cat_24",
    },
    {
        "name": "categorical-feature-25",
        "vocabulary_size": 405282,
        "feature_list_length": 8,
        "new_name": "cat_25",
    },
    {
        "name": "categorical-feature-26",
        "vocabulary_size": 10,
        "feature_list_length": 1,
        "new_name": "cat_26",
    },
    {
        "name": "categorical-feature-27",
        "vocabulary_size": 2209,
        "feature_list_length": 6,
        "new_name": "cat_27",
    },
    {
        "name": "categorical-feature-28",
        "vocabulary_size": 11938,
        "feature_list_length": 9,
        "new_name": "cat_28",
    },
    {
        "name": "categorical-feature-29",
        "vocabulary_size": 155,
        "feature_list_length": 5,
        "new_name": "cat_29",
    },
    {
        "name": "categorical-feature-30",
        "vocabulary_size": 4,
        "feature_list_length": 1,
        "new_name": "cat_30",
    },
    {
        "name": "categorical-feature-31",
        "vocabulary_size": 976,
        "feature_list_length": 1,
        "new_name": "cat_31",
    },
    {
        "name": "categorical-feature-32",
        "vocabulary_size": 14,
        "feature_list_length": 1,
        "new_name": "cat_32",
    },
    {
        "name": "categorical-feature-33",
        "vocabulary_size": 40000000,
        "feature_list_length": 12,
        "new_name": "cat_33",
    },
    {
        "name": "categorical-feature-34",
        "vocabulary_size": 40000000,
        "feature_list_length": 100,
        "new_name": "cat_34",
    },
    {
        "name": "categorical-feature-35",
        "vocabulary_size": 40000000,
        "feature_list_length": 27,
        "new_name": "cat_35",
    },
    {
        "name": "categorical-feature-36",
        "vocabulary_size": 590152,
        "feature_list_length": 10,
        "new_name": "cat_36",
    },
    {
        "name": "categorical-feature-37",
        "vocabulary_size": 12973,
        "feature_list_length": 3,
        "new_name": "cat_37",
    },
    {
        "name": "categorical-feature-38",
        "vocabulary_size": 108,
        "feature_list_length": 1,
        "new_name": "cat_38",
    },
    {
        "name": "categorical-feature-39",
        "vocabulary_size": 36,
        "feature_list_length": 1,
        "new_name": "cat_39",
    },
]

# === Model ===
model_config = Config()
# Embedding
model_config.embedding_dim = 128
model_config.allow_id_dropping = True
model_config.embedding_threshold = 21000
model_config.max_ids_per_partition = 8192
model_config.max_unique_ids_per_partition = 4096
model_config.learning_rate = 0.0034

# MLP
model_config.bottom_mlp_dims = [512, 256, 128]
model_config.top_mlp_dims = [1024, 1024, 512, 256, 1]

# DCN
model_config.num_dcn_layers = 3
model_config.dcn_projection_dim = 512

# === Training ===
training_config = Config()
training_config.learning_rate = 0.0034
training_config.global_batch_size = 128
# Set `num_steps` in the main config file instead of num_epochs, because we are
# using a Python generator.
training_config.num_steps = 2
training_config.eval_freq = 1

# === Assign all configs to the root config ===
config.dataset = dataset_config
config.model = model_config
config.training = training_config

config.freeze()
