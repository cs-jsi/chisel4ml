/*
 * HEADER: TODO 
 *
 *
 * This file contains the definition of the Model.
 */

package chisel4ml

import chisel3._
import chisel3.util._
import chisel3.experimental._
import scala.collection._

object ProcessingPipeline (model : lbir.Model) extends Module {  
  val inSize = model.layers(0).inSize
  val inSizeBits = hwLayers(0).inSizeBits
  val outSize  = hwLayers.last.outSize

  val io = IO(new Bundle {
      val in  = Input(UInt(inSizeBits.W))
      val out = Output(UInt(outSize.W))
  })

  // List of processing elements - one pe per layer
  val peList = new mutable.MutableList[Layer]() 
  
  for (layer <- model.layers) {
      assert(layer.ltype == lbir.Layer.Type.DENSE)
      peList += Module(new ProcessingElementFactory(layer))
  }


  peList(0).io.in := io.in
  for (i <- 1 until hwLayers.length) { 
      assert(peList(i).inSize == peList(i-1).outSize)
      peList(i).io.in := peList(i-1).io.out
    }
  io.out := hwLayers.last.io.out
}


