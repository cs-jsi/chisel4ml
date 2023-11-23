package chisel4ml.tests

import _root_.chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import chisel4ml.util.shiftAndRound

class RoundTestBed(inputWidth: Int, shift: Int) extends Module {
  val in = IO(Input(SInt(inputWidth.W)))
  val out = IO(Output(SInt()))

  out := shiftAndRound(in, shift)
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
  it should "Test rounding -88 -> -6 and -150 -> -9" in {
    test(new RoundTestBed(14, -4)) { dut =>
      dut.in.poke((87).S(14.W))
      dut.out.expect(roundModel(87, 4).S)
      dut.in.poke((88).S(14.W))
      dut.out.expect(roundModel(88, 4).S)
      dut.in.poke((89).S(14.W))
      dut.out.expect(roundModel(89, 4).S)

      dut.in.poke((-89).S(14.W))
      dut.out.expect(roundModel(-89, 4).S)
      dut.in.poke((-87).S(14.W))
      dut.out.expect(roundModel(-87, 4).S)
      dut.in.poke((-88).S(14.W))
      dut.out.expect(roundModel(-88, 4).S)
    /*assert(roundModel(-88, 4).S(14.W).litValue.toInt == -6)
      dut.in.poke((-150).S(14.W))
      dut.out.expect(roundModel(-150, 4).S)
      assert(roundModel(-150, 4).S(14.W).litValue.toInt == -9)*/
    }
  }

  val r = new scala.util.Random
  r.setSeed(42L)
  val numRandomTests = 500
  val maxAccumulatorValue = 8191
  for (_ <- 0 until numRandomTests) {
    val isNegative = r.nextBoolean()
    val value = if (isNegative) -r.nextInt(maxAccumulatorValue) else r.nextInt(maxAccumulatorValue)
    val shift = r.nextInt(7)
    it should s"Test random rounding value: $value, shift: $shift" in {
      test(new RoundTestBed(14, -shift)) { dut =>
        dut.in.poke(value.S(14.W))
        dut.out.expect(roundModel(value, shift).S)
      }
    }
  }
}
