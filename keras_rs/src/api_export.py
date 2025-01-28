from typing import Any, TypeVar

import keras

try:
    import namex
except ImportError:
    namex = None


T = TypeVar("T", bound=Any)


class keras_rs_export:
    def __init__(self, path: str):
        if namex is not None:
            self.namex_export = namex.export(package="keras_rs", path=path)

    def __call__(self, symbol: T) -> T:
        keras.saving.register_keras_serializable(package="keras_rs")(symbol)
        if namex is not None:
            self.namex_export(symbol)
        return symbol
