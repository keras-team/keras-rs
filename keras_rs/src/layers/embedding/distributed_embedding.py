import keras

from keras_rs.src.api_export import keras_rs_export

if keras.backend.backend() == "jax":
    from keras_rs.src.layers.embedding.jax.distributed_embedding import (
        DistributedEmbedding as BackendDistributedEmbedding,
    )
elif keras.backend.backend() == "tensorflow":
    from keras_rs.src.layers.embedding.tensorflow.distributed_embedding import (
        DistributedEmbedding as BackendDistributedEmbedding,
    )
else:
    from keras_rs.src.layers.embedding.base_distributed_embedding import (
        DistributedEmbedding as BackendDistributedEmbedding,
    )


@keras_rs_export("keras_rs.layers.DistributedEmbedding")
class DistributedEmbedding(BackendDistributedEmbedding):
    pass
