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
import scala.collection.mutable._

class ProcessingPipeline (model : lbir.Model) extends Module {  
  // List of processing elements - one PE per layer
  val peList = new ListBuffer[ProcessingElementSimple]() 
 
  // Instantiate modules for seperate layers, for now we only support DENSE layers
  for (layer <- model.layers) {
      assert(layer.ltype == lbir.Layer.Type.DENSE)
      //peList += Module(new ProcessingElementFactory(layer))
      peList += Module(new BinarizedDense(layer))
  }

  val io = IO(new Bundle {
      val in  = Input(UInt(peList(0).inSizeBits.W))
      val out = Output(UInt(peList.last.outSizeBits.W))
  })

  // Connect the inputs and outputs of the layers
  peList(0).io.in := io.in
  for (i <- 1 until model.layers.length) { 
      assert(peList(i).inSize == peList(i-1).outSize)
      peList(i).io.in := peList(i-1).io.out
    }
  io.out := peList.last.io.out
}
