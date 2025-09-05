from keras.utils import Config

# === Model ===
model_config = Config()
# Embedding
model_config.embedding_dim = 128
model_config.allow_id_dropping = True
model_config.embedding_threshold = 21000
model_config.max_ids_per_partition = 4096
model_config.max_unique_ids_per_partition = 2048
model_config.learning_rate = 0.005

# MLP
model_config.bottom_mlp_dims = [512, 256, 128]
model_config.top_mlp_dims = [1024, 1024, 512, 256, 1]

# DCN
model_config.num_dcn_layers = 3
model_config.dcn_projection_dim = 512
