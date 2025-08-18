from keras.utils import Config

from .datasets.dummy_dataset import dataset_config
from .models.default_model import model_config
from .training.default_training import training_config

config = Config()

config.experiment_name = "v6e_8_full_dataset"
config.model_dir = "./v6e_8_full_dataset"

config.dataset = dataset_config
config.dataset.file_pattern = (
    "gs://qinyiyan-vm/mlperf-dataset/criteo_merge_balanced_4224/"
    "train-00000-of-01024tfrecord"
)
config.model = model_config
config.training = training_config
config.training.batch_size = 4224

config.freeze()
