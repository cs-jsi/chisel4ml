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
        require(args.size == 2)
        val genDir = Paths.get(args(0))
        val byteArray = Files.readAllBytes(Paths.get(args(1)))
        val lbirModel = lbir.Model.parseFrom(byteArray)
        (new ChiselStage).emitVerilog(new ProcessingPipeline(lbirModel), 
                                      Array("-td", genDir.toAbsolutePath().toString()))
    }
}
