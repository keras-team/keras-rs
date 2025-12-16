from keras.utils import Config

config = Config()

# === Experiment metadata ===
config.experiment_name = "v6e_16"
config.model_dir = "./v6e_16"

# === Dataset ===
dataset_config = Config()
dataset_config.file_pattern = None
dataset_config.val_file_pattern = None

# Features
dataset_config.label = "clicked"
dataset_config.dense = [f"int-feature-{i}" for i in range(1, 14)]
dataset_config.lookup = [
    {
        "name": "categorical-feature-14",
        "vocabulary_size": 40000000,
        "feature_list_length": 3,
        "new_name": "0",
    },
    {
        "name": "categorical-feature-15",
        "vocabulary_size": 39060,
        "feature_list_length": 2,
        "new_name": "1",
    },
    {
        "name": "categorical-feature-16",
        "vocabulary_size": 17295,
        "feature_list_length": 1,
        "new_name": "2",
    },
    {
        "name": "categorical-feature-17",
        "vocabulary_size": 7424,
        "feature_list_length": 2,
        "new_name": "3",
    },
    {
        "name": "categorical-feature-18",
        "vocabulary_size": 20265,
        "feature_list_length": 6,
        "new_name": "4",
    },
    {
        "name": "categorical-feature-19",
        "vocabulary_size": 3,
        "feature_list_length": 1,
        "new_name": "5",
    },
    {
        "name": "categorical-feature-20",
        "vocabulary_size": 7122,
        "feature_list_length": 1,
        "new_name": "6",
    },
    {
        "name": "categorical-feature-21",
        "vocabulary_size": 1543,
        "feature_list_length": 1,
        "new_name": "7",
    },
    {
        "name": "categorical-feature-22",
        "vocabulary_size": 63,
        "feature_list_length": 1,
        "new_name": "8",
    },
    {
        "name": "categorical-feature-23",
        "vocabulary_size": 40000000,
        "feature_list_length": 7,
        "new_name": "9",
    },
    {
        "name": "categorical-feature-24",
        "vocabulary_size": 3067956,
        "feature_list_length": 3,
        "new_name": "10",
    },
    {
        "name": "categorical-feature-25",
        "vocabulary_size": 405282,
        "feature_list_length": 8,
        "new_name": "11",
    },
    {
        "name": "categorical-feature-26",
        "vocabulary_size": 10,
        "feature_list_length": 1,
        "new_name": "12",
    },
    {
        "name": "categorical-feature-27",
        "vocabulary_size": 2209,
        "feature_list_length": 6,
        "new_name": "13",
    },
    {
        "name": "categorical-feature-28",
        "vocabulary_size": 11938,
        "feature_list_length": 9,
        "new_name": "14",
    },
    {
        "name": "categorical-feature-29",
        "vocabulary_size": 155,
        "feature_list_length": 5,
        "new_name": "15",
    },
    {
        "name": "categorical-feature-30",
        "vocabulary_size": 4,
        "feature_list_length": 1,
        "new_name": "16",
    },
    {
        "name": "categorical-feature-31",
        "vocabulary_size": 976,
        "feature_list_length": 1,
        "new_name": "17",
    },
    {
        "name": "categorical-feature-32",
        "vocabulary_size": 14,
        "feature_list_length": 1,
        "new_name": "18",
    },
    {
        "name": "categorical-feature-33",
        "vocabulary_size": 40000000,
        "feature_list_length": 12,
        "new_name": "19",
    },
    {
        "name": "categorical-feature-34",
        "vocabulary_size": 40000000,
        "feature_list_length": 100,
        "new_name": "20",
    },
    {
        "name": "categorical-feature-35",
        "vocabulary_size": 40000000,
        "feature_list_length": 27,
        "new_name": "21",
    },
    # {
    #     "name": "categorical-feature-36",
    #     "vocabulary_size": 590152,
    #     "feature_list_length": 10,
    #     "new_name": "22",
    # },
    {
        "name": "categorical-feature-37",
        "vocabulary_size": 12973,
        "feature_list_length": 3,
        "new_name": "23",
    },
    {
        "name": "categorical-feature-38",
        "vocabulary_size": 108,
        "feature_list_length": 1,
        "new_name": "24",
    },
    {
        "name": "categorical-feature-39",
        "vocabulary_size": 36,
        "feature_list_length": 1,
        "new_name": "25",
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
training_config.global_batch_size = 16896
# Set `num_steps` instead of `num_epochs`, because we are using a Python
# generator.
training_config.num_steps = 10
training_config.eval_freq = 5
training_config.num_eval_steps = 10

# === Assign all configs to the root config ===
config.dataset = dataset_config
config.model = model_config
config.training = training_config

config.freeze()
