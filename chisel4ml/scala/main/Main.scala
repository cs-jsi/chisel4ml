/*
 * HEADER: TODO
 *
 */
package chisel4ml

import java.nio.file.{Files, Paths}
import chisel3.stage.ChiselStage
import chisel3._

/**
 * An object extending App to generate the Verilog code.
 */
object Main {

    def main(args: Array[String]): Unit = {
        val byteArray = Files.readAllBytes(Paths.get(args(0)))
        val lbirModel = lbir.Model.parseFrom(byteArray)
        (new ChiselStage).emitVerilog(Module(new ProcessingPipeline(lbirModel)), 
                                      Array("-td","gen/"))
    }
}
