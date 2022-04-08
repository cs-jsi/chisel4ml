from tensorflow.keras.layers import Layer as KerasLayer

from typing import List

from chisel4ml.optimizations.qkeras_optimization import QKerasOptimization


class QKerasDefaultOptimization(QKerasOptimization):
    num_layers = 1

    def __call__(self, layers: List[KerasLayer]) -> List[KerasLayer]:
        return layers
