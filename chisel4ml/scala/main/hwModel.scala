/*
 * HEADER: TODO 
 *
 *
 * This file contains the definition of the Model.
 */

package mpbnn_hw_gen

import chisel3._
import chisel3.util._
import chisel3.experimental._
import scala.collection._

object hwModel (model : lbir.Model) extends Module {  

  val hwLayers = new mutable.MutableList[Layer]() 
  
  // This code should eventually become superfluous, the plan is to add an IR
  // that will already represent the conversion bellow. The hardware layers will 
  // then simply be parameterized by software layers.
  for (swLayer <- model.config.layers) {
      assert(swLayer.class_name == "QuantDense")
      
      swLayer.config.asInstanceOf[QuantDenseLayerConfig].
      input_quantizer.class_name match {
        case "SteSign" => {
          val weights : Array[Array[Bool]] = swLayer.config.asInstanceOf[QuantDenseLayerConfig].
                                             kernel.map(_.map( x => (x > 0).B ))
          val thresh : Array[UInt] = swLayer.config.asInstanceOf[QuantDenseLayerConfig].
                                     thresh.map( x => x.toInt.U )
          
          hwLayers += Module(new BinarizedDense(weights.transpose, thresh))
        }
        case "" => {
          val weights : Array[Array[Bool]] = swLayer.config.asInstanceOf[QuantDenseLayerConfig].
                                             kernel.map(_.map( x => (x > 0).B ))
          val thresh : Array[FixedPoint] = swLayer.config.asInstanceOf[QuantDenseLayerConfig].
                                     thresh.map( x => FixedPoint.fromDouble(x, 13.W, 7.BP) )
          
          hwLayers += Module(new FixedPointDense(weights.transpose, thresh))
        }
      }
  }
  

  val inSize = hwLayers(0).inSize
  val inSizeBits = hwLayers(0).inSizeBits
  val outSize  = hwLayers.last.outSize

  val io = IO(new Bundle {
      val in  = Input(UInt(inSizeBits.W))
      val out = Output(UInt(outSize.W))
  })

  hwLayers(0).io.in := io.in
  for (i <- 1 until hwLayers.length) { 
      assert(hwLayers(i).inSize == hwLayers(i-1).outSize)
      hwLayers(i).io.in := hwLayers(i-1).io.out
    }
  io.out := hwLayers.last.io.out
}


