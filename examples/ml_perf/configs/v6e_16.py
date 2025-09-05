from keras.utils import Config

from .datasets.dummy_dataset import dataset_config
from .models.default_model import model_config
from .training.default_training import training_config

config = Config()

config.experiment_name = "v6e_16"
config.model_dir = "./v6e_16"

config.dataset = dataset_config
config.model = model_config
config.training = training_config

config.freeze()
