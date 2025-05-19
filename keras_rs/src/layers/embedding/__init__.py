from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.layers.embedding.base_distributed_embedding import (
    DistributedEmbedding as BackendDistributedEmbedding,
)


@keras_rs_export("keras_rs.layers.DistributedEmbedding")
class DistributedEmbedding(BackendDistributedEmbedding):
    pass
