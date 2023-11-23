package chisel4ml.tests

import _root_.chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import chisel4ml.util.shiftAndRoundSIntStatic

class RoundTestBed(inputWidth: Int, shift: Int) extends Module {
  val in = IO(Input(SInt(inputWidth.W)))
  val out = IO(Output(SInt()))

  //out := shiftAndRoundSInt(in, shift.abs.U, (shift >= 0).B)
  out := shiftAndRoundSIntStatic(in, shift)
}

object UtilityFunctionsTests {
  // This function models the rounding function in the quantized_bits quantization
  // operator of QKeras.
  def roundModel(value: Int, shift: Int): Int = {
    val scale = Math.pow(2.toDouble, shift.toDouble)
    (math.floor((value.abs / scale) + 0.5) * value.sign).toInt
  }
}

class UtilityFunctionsTests extends AnyFlatSpec with ChiselScalatestTester {
  behavior.of("utilities")
  import UtilityFunctionsTests.roundModel
  for (number <- Seq(87, 88, 89, -87, -88, -89, -150)) {
    it should s"Test rounding $number with shift == 4. Should be: ${roundModel(number, 4)}" in {
      test(new RoundTestBed(14, -4)) { dut =>
        dut.in.poke(number.S(14.W))
        dut.out.expect(roundModel(number, 4).S)
      }
    }
  }

  val r = new scala.util.Random
  r.setSeed(42L)
  val numRandomTests = 10
  val maxAccumulatorValue = 8191
  for (idx <- 0 until numRandomTests) {
    val isNegative = r.nextBoolean()
    val value = if (isNegative) -r.nextInt(maxAccumulatorValue) else r.nextInt(maxAccumulatorValue)
    val shift = 4
    it should s"Random test $idx rounding value: $value, shift: 4" in {
      test(new RoundTestBed(14, -shift)) { dut =>
        dut.in.poke(value.S(14.W))
        dut.out.expect(roundModel(value, shift).S)
      }
    }
  }
}
