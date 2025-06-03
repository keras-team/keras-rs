from keras_rs.src.api_export import keras_rs_export

# Unique source of truth for the version number.
__version__ = "0.2.1"


@keras_rs_export("keras_rs.version")
def version() -> str:
    return __version__
