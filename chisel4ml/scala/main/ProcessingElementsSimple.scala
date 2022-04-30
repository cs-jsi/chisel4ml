/*
 * HEADER: TODO 
 *
 *
 * This file contains the definition of the abstract class ProcessingElement.
 */

package chisel4ml

import chisel3._
import chisel3.util._
import chisel3.experimental._

/**
 *  Base class of all ProcessingElements.
 *
 *  The base class of all ProcessingElements.:
 *
 * @param layer Is a layer definition defined by the LBIR format.
 */
abstract class ProcessingElementSimple(layer: lbir.Layer) extends Module {
  /**
   * Number of inputs.
   */
  val inSize: Int = layer.input.get.shape.reduce(_ * _)  

   /**
   * Number of input bits.
   */
  val inSizeBits: Int = inSize * layer.input.get.dtype.get.bitwidth
  
  /**
   * Number of outputs.
   */
  val outSize: Int = layer.outShape.reduce(_ * _)

  /**
   * outSizeBits is the number of output bits. It determines the width of
   * the outgoing UInt.
   */
  val outSizeBits: Int = outSize * layer.activation.get.bitwidth

  
  /*
   * Determines the input output interface. This gets cast to various data
   * types in Layer implementation classes.
   */  
  val io = IO(new Bundle {
      val in  = Input(UInt(inSizeBits.W))
      val out = Output(UInt(outSizeBits.W))
  })
}

/**
 * The BinarizedDense layer implements a fully-connected binarized layer.
 *
 * This layer implements a fully-connected layer as specified in the paper 
 * Hubara et al.: Binarized Neural Networks (https://arxiv.org/abs/1602.02830).
 *   
 */
class BinarizedDense(lbirLayer: lbir.Layer) extends ProcessingElementSimple(lbirLayer) {
  // We import the values as UInts and the convert them to Bool Vectors, because
  // in Verilog this results as an Array, instead of a number of individual elements
  // in the interface. (I.e. in[0:2] instead of in_0, in_1 and in_2.)
  val in_int = Wire(Vec(inSize, Bool()))
  val out_int = Wire(Vec(outSize, Bool()))

  val weights: Seq[Seq[Bool]] = lbirLayer.weights.get.values.map(_ > 0).map(_.B).sliding(inSize, inSize).toSeq
  val thresh: Seq[UInt] = lbirLayer.biases.get.values.map(_.toInt.U)
    
  in_int := io.in.asTypeOf(in_int)  

  def binarizedNeuron(in:Seq[Bool], weights:Seq[Bool], thresh:UInt) : Bool = {
    require(weights.length == in.length)
    val act = PopCount((in zip weights).map{ case(a:Bool, b:Bool) => ~(a ^ b) } )
    act > thresh
  }

  for (i <- 0 until outSize) { 
    out_int(i) := binarizedNeuron(in_int, weights(i), thresh(i))
  }

  // The CAT operator reverses the order of bits, so we reverse them 
  // to evenout the reversing (its not pretty but it works).
  io.out := Cat(out_int.reverse)
}

/*
 * This is typically an input layer. As described in paper Hubara et al.: Binarized Neural
 * Networks (https://arxiv.org/abs/1602.02830), the first layer of a BNN should no have binarized
 * inputs, as this tends to lead to significantly lower performance. This layer still has binarized
 * weights, but the MAC computation is done in FixedPoint. Luckily, as described in the paper above,
 * this can be done with a serie of XNOR gates.
 *
 *  Example arithmetic:
 *  0.75 * 1 
 *    ->      0110 0000 (0.75 in S(0,8) FixedPoint format)
 *    -> XNOR 1111 1111
 *    ->    = 0110 0000 = 2^-1 +2^-2 = 0.75 (0.75 * 1 does indeed equal to 1)
 *  0.75 * (-1)
 *    ->      0110 0000
 *    -> XNOR 0000 0000 (We again use 0 to repesent -1)
 *    ->    = 1001 1111 = -2^0 + 2^-3 + 2^-5 + 2^-6 + 2^-7 = -0.75781 or about -0.75
 *
 *  -> This than needs to be added together and compared with the threshold.
 */
/*class FixedPointDense(lbirLayer:lbir.Layer) extends ProcessingElementSimple(lbirLayer) {
  val numTotalBits:Int = thresh(0).getWidth
  val numFractBits = thresh(0).binaryPoint 

  // We import the values as UInts and the convert them to Bool Vectors, because
  // in Verilog this results as an Array, instead of a number of individual elements
  // in the interface. (I.e. in[0:2] instead of in_0, in_1 and in_2.)
  // The input data type should be the same as the threshold.
  val in_int = Wire(Vec(inSize, FixedPoint(numTotalBits.W, numFractBits))) 
  val out_int = Wire(Vec(outSize, Bool()))

  // This function approximates the multiplication with 1 and -1. Because these
  // numbers are in twos complement, to get -1*a you just need to invert the bits
  // and add 1. We do not add this extra ONE to save on FPGA area. WARNING!
  // TODO: remove this LSB value from thresh, then the computation will be equivalent
  def multiplyXNOR(inFp:FixedPoint, inBool: Bool): FixedPoint = {
    Mux(inBool, inFp, inFp.asUInt().unary_~().asFixedPoint(inFp.binaryPoint))
  }  

  def binarizedNeuron(in:Seq[FixedPoint], weights:Seq[Bool], thresh:FixedPoint): Bool = {
    require(weights.length == in.length)
    val act = (in zip weights).map{ case(a:FixedPoint, b:Bool) => multiplyXNOR(a, b) }.reduce(_+_)
    act > thresh
  }

  in_int := io.in.asTypeOf(in_int)
  for (i <- 0 until outSize) {
    out_int(i) := binarizedNeuron(in_int, weights(i), thresh(i))
  }

  // The CAT operator reverses the order of bits, so we reverse them 
  // to evenout the reversing (its not pretty but it works).
  io.out := Cat(out_int.reverse)
}*/

