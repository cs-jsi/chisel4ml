/*
 * HEADER: TODO
 *
 */
package chisel4ml

import _root_.java.nio.file.{Files, Paths}
import _root_.scala.concurrent.{ExecutionContext, Future}

import _root_.chisel3.stage._
import _root_.chisel3._

import _root_.io.grpc.{Server, ServerBuilder}
import _root_.services.{PpServiceGrpc, 
                        PpRunReturn, 
                        PpRunParams, 
                        PpElaborateReturn, 
                        PpHandle, 
                        ErrorMsg,
                        GenerateParams}
import _root_.lbir.{Model, QTensor, Datatype}

import _root_.treadle.TreadleTester
import _root_.org.slf4j.Logger
import _root_.org.slf4j.LoggerFactory


/**
 * An object extending App to generate the Verilog code.
 */
object Chisel4mlServer {
    private val port = 50051
 
    def main(args: Array[String]): Unit = {
        val server = new Chisel4mlServer(ExecutionContext.global)
        server.start()
        server.blockUntilShutdown()
    }
}

class Chisel4mlServer(executionContext: ExecutionContext) { self =>
    private[this] var server: Server = null
    val logger = LoggerFactory.getLogger(classOf[Chisel4mlServer])

    private def start(): Unit = {
      server = ServerBuilder.forPort(Chisel4mlServer.port).addService(
            PpServiceGrpc.bindService(new PpServiceImpl, executionContext)).build.start
      sys.addShutdownHook {
        self.stop()
      }
      logger.info("Started a new chisel4ml server.")
    }

    private def stop(): Unit = {
      if (server != null) {
        logger.info("Shutting down chisel4ml server.")
        server.shutdown()
      }
      else {
        logger.error("Attempted to shut down server that was not created.")
      }
    }

    private def blockUntilShutdown(): Unit = {
      if (server != null) {
        server.awaitTermination()
      }
    } 

    def toBinary(i: Int, digits: Int = 8) =
        String.format("%" + digits + "s", i.toBinaryString).replace(' ', '0')

    def toBinaryB(i: BigInt, digits: Int = 8) =
        String.format("%" + digits + "s", i.toString(2)).replace(' ', '0')

    // TODO move this function somewhere else
    def lbirToBigInt(qtensor: QTensor): BigInt = {
        var values = qtensor.values.reverse
        if (qtensor.dtype.get.quantization == Datatype.QuantizationType.BINARY) {
            values = values.map(x => (x + 1) / 2) // 1 -> 1, -1 -> 0
        }
        
        val string_int = values.map(x => toBinary(x.toInt, qtensor.dtype.get.bitwidth)).mkString 
        val big_int = BigInt(string_int, radix=2)
        logger.info("Converted lbir.QTensor: " + qtensor.values + " to BigInt: " + string_int + "." + 
                    " The number of bits is: " + qtensor.dtype.get.bitwidth + ".")
        big_int
    }

    def bigIntToLbir(value: BigInt, outSize: Int): QTensor = {
        val dataType = Datatype(quantization=Datatype.QuantizationType.BINARY,
                                bitwidth=1,
                                scale=1,
                                offset=0)
        // We substract the 48 because x is an ASCII encoded symbol
        val lbir_values = toBinaryB(value, outSize).toList.map(x=>x.toFloat-48).reverse.map(x => (x * 2) -1) 
        val qtensor = QTensor(dtype=Option(dataType), shape = List(outSize), values=lbir_values)
        logger.info("Converted BigInt: " + value + " to lbir.QTensor: " + qtensor + 
                    ". The number of bits is " + outSize + ".")
        qtensor
    }

    private object PpServiceImpl {
        var model: Model = null
        var tester: TreadleTester = null
    }

    private class PpServiceImpl extends PpServiceGrpc.PpService {
        private[this] var model: Model = null
        private[this] var tester: TreadleTester = null

        override def elaborate(lbirModel: Model): Future[PpElaborateReturn] = {
            logger.info("Elaborating model: " + lbirModel.name + " to a processing pipeline circuit.")
            model = lbirModel
            tester = TreadleTester((new ChiselStage).execute(Array(), 
                                    Seq(ChiselGeneratorAnnotation(() => new ProcessingPipeline(lbirModel)))))

            val errReply = ErrorMsg(err = ErrorMsg.ErrorId.SUCCESS, msg="Everything went fine.")
            val ppHandle = PpHandle(name = "model", 
                                    input=lbirModel.layers(0).input,
                                    outShape=lbirModel.layers.last.outShape)
            val ppElaborateReturn = PpElaborateReturn(ppHandle = Option(ppHandle),
                                                      reply = Some(errReply))
            Future.successful(ppElaborateReturn)
        }

        override def run(ppRunParams: PpRunParams): Future[PpRunReturn] = {
            logger.info("Simulating processing pipeline: " + ppRunParams.ppHandle.get.name + " circuit on inputs.")
            tester.poke("io_in", lbirToBigInt(ppRunParams.inputs(0)))
            Future.successful(PpRunReturn(values = List(bigIntToLbir(tester.peek("io_out"), 
                                                                     model.layers.last.outShape(0)))))
        }

        override def generate(genParams: GenerateParams): Future[ErrorMsg] = {
            logger.info("Generating hardware for circuit id: " + genParams.name + " in directory: " + 
                         genParams.directory + ".")
            (new ChiselStage).execute(Array("--target-dir", genParams.directory), 
                                      Seq(ChiselGeneratorAnnotation(() => new ProcessingPipeline(model))))
            Future.successful(ErrorMsg(err=ErrorMsg.ErrorId.SUCCESS, msg="Successfully generated verilog"))
        }
    }
}
