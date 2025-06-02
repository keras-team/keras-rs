import importlib.util
import platform
import sys

import keras

from keras_rs.src.api_export import keras_rs_export

# JAX distributed embedding is only available on linux_x86_64, and only if
# jax-tpu-embedding is installed.
jax_tpu_embedding_spec = importlib.util.find_spec("jax_tpu_embedding")
if (
    keras.backend.backend() == "jax"
    and sys.platform == "linux"
    and platform.machine().lower() == "x86_64"
    and jax_tpu_embedding_spec is not None
):
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
