from chisel4ml.preprocess.fft_qonnx_op import FFTreal
from chisel4ml.preprocess.lmfe_qonnx_op import lmfe
from chisel4ml.qtensor import QTensor

custom_op = {"FFTreal": FFTreal, "qtensor": QTensor, "lmfe": lmfe}
