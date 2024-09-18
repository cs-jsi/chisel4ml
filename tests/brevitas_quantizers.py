from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject.enum import BitWidthImplType
from brevitas.inject.enum import FloatToIntImplType
from brevitas.inject.enum import QuantType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import StatsOp
from brevitas.quant.solver import ActQuantSolver
from brevitas.quant.solver import BiasQuantSolver
from brevitas.quant.solver import WeightQuantSolver


class LearnedSFInt4WeightPerChannel(WeightQuantSolver):
    quant_type = QuantType.INT  # integer quantization
    bit_width_impl_type = BitWidthImplType.CONST  # constant bit width
    float_to_int_impl_type = FloatToIntImplType.ROUND  # round to nearest
    scaling_impl_type = (
        ScalingImplType.PARAMETER_FROM_STATS
    )  # scale based on statistics
    scaling_stats_op = StatsOp.MAX  # scale statistics is the absmax value
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    scaling_per_output_channel = True
    bit_width = 4
    signed = True  # quantization range is signed
    narrow_range = False  # quantization range is [-127,127] rather than [-128, 127]
    zero_point_impl = ZeroZeroPoint  # zero point is 0.


class Int8BiasQuant(BiasQuantSolver):
    quant_type = QuantType.INT  # integer quantization
    bit_width_impl_type = BitWidthImplType.CONST  # constant bit width
    float_to_int_impl_type = FloatToIntImplType.ROUND  # round to nearest
    scaling_impl_type = ScalingImplType.PARAMETER
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    requires_input_bit_width = False
    requires_input_scale = False
    scaling_per_output_channel = False
    bit_width = 8
    scaling_init = 2 ** (bit_width - 1)
    signed = True  # quantization range is signed
    narrow_range = False  # quantization range is [-127,127] rather than [-128, 127]
    zero_point_impl = ZeroZeroPoint  # zero point is 0.


class Int4ActQuant(ActQuantSolver):
    quant_type = QuantType.INT  # integer quantization
    bit_width_impl_type = BitWidthImplType.CONST  # constant bit width
    float_to_int_impl_type = FloatToIntImplType.ROUND  # round to nearest
    scaling_impl_type = (
        ScalingImplType.CONST
    )  # scale is a parameter initialized from statistics
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    scaling_per_output_channel = False  # scale is per tensor
    bit_width = 4
    signed = True  # quantization range is signed
    narrow_range = False  # quantization range is [-128, 127] rather than [-127, 127]
    zero_point_impl = ZeroZeroPoint  # zero point is 0.
    scaling_init = 2 ** (bit_width - 1)


class Int8ActQuant(ActQuantSolver):
    quant_type = QuantType.INT  # integer quantization
    bit_width_impl_type = BitWidthImplType.CONST  # constant bit width
    float_to_int_impl_type = FloatToIntImplType.ROUND  # round to nearest
    scaling_impl_type = (
        ScalingImplType.CONST
    )  # scale is a parameter initialized from statistics
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    scaling_per_output_channel = False  # scale is per tensor
    bit_width = 8  # bit width is 8
    signed = True  # quantization range is signed
    narrow_range = False  # quantization range is [-128, 127] rather than [-127, 127]
    zero_point_impl = ZeroZeroPoint  # zero point is 0.
    scaling_init = 2 ** (bit_width - 1)


class Int12ActQuant(ActQuantSolver):
    quant_type = QuantType.INT  # integer quantization
    bit_width_impl_type = BitWidthImplType.CONST  # constant bit width
    float_to_int_impl_type = FloatToIntImplType.ROUND  # round to nearest
    scaling_impl_type = (
        ScalingImplType.CONST
    )  # scale is a parameter initialized from statistics
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    scaling_per_output_channel = False  # scale is per tensor
    bit_width = 12
    signed = True  # quantization range is signed
    narrow_range = False  # quantization range is [-128, 127] rather than [-127, 127]
    zero_point_impl = ZeroZeroPoint  # zero point is 0.
    scaling_init = 2 ** (bit_width - 1)


class Int31ActQuant(ActQuantSolver):
    quant_type = QuantType.INT  # integer quantization
    bit_width_impl_type = BitWidthImplType.CONST  # constant bit width
    float_to_int_impl_type = FloatToIntImplType.ROUND  # round to nearest
    scaling_impl_type = (
        ScalingImplType.CONST
    )  # scale is a parameter initialized from statistics
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    scaling_per_output_channel = False  # scale is per tensor
    bit_width = 31
    signed = True  # quantization range is signed
    narrow_range = False  # quantization range is [-128, 127] rather than [-127, 127]
    zero_point_impl = ZeroZeroPoint  # zero point is 0.
    scaling_init = 2 ** (bit_width - 1)


class Int32ActQuant(ActQuantSolver):
    quant_type = QuantType.INT  # integer quantization
    bit_width_impl_type = BitWidthImplType.CONST  # constant bit width
    float_to_int_impl_type = FloatToIntImplType.ROUND  # round to nearest
    scaling_impl_type = (
        ScalingImplType.CONST
    )  # scale is a parameter initialized from statistics
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    scaling_per_output_channel = False  # scale is per tensor
    bit_width = 32
    signed = True  # quantization range is signed
    narrow_range = False  # quantization range is [-128, 127] rather than [-127, 127]
    zero_point_impl = ZeroZeroPoint  # zero point is 0.
    scaling_init = 2 ** (bit_width - 1)


class Int33ActQuant(ActQuantSolver):
    quant_type = QuantType.INT  # integer quantization
    bit_width_impl_type = BitWidthImplType.CONST  # constant bit width
    float_to_int_impl_type = FloatToIntImplType.ROUND  # round to nearest
    scaling_impl_type = (
        ScalingImplType.CONST
    )  # scale is a parameter initialized from statistics
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    scaling_per_output_channel = False  # scale is per tensor
    bit_width = 33
    signed = True  # quantization range is signed
    narrow_range = False  # quantization range is [-128, 127] rather than [-127, 127]
    zero_point_impl = ZeroZeroPoint  # zero point is 0.
    scaling_init = 2 ** (bit_width - 1)
