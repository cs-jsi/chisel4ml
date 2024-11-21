/*
 * Copyright 2022 Computer Systems Department, Jozef Stefan Insitute
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package chisel4ml.logging
import chisel4ml._
import lbir.LayerWrap
import org.chipsalliance.cde.config.{Field, Parameters}
import org.slf4j.LoggerFactory

trait HasLogger {
  val logger = LoggerFactory.getLogger(this.getClass().toString())
}

trait HasParameterLogging extends HasLogger {
  // Macro to find this?
  private def fields: Seq[Field[_]] = Seq(
    LayerWrapSeqField,
    NumBeatsInField,
    NumBeatsOutField
  )

  def logParameters(
    implicit p: Parameters
  ): Unit = {
    var msg = s"Generated new ${this.getClass()} module.\n"
    for (field <- fields) {
      try {
        val pValue = p(field)
        val pName = field.getClass().getSimpleName()
        val str = pValue match {
          case l: LayerWrap =>
            s""" Input shape: ${l.input.shape},
               | Input quantization: ${l.input.dtype.quantization},
               | Input sign: ${l.input.dtype.signed},
               | Input shift: ${l.input.dtype.shift},
               | Output shape: ${l.output.shape},
               | Output quantization: ${l.output.dtype.quantization},
               | Output sign: ${l.output.dtype.signed},
               | Output shift: ${l.output.dtype.shift}
               | Other parameters are: """.stripMargin
          case _ => s"$pName->$pValue, "
        }
        msg = msg + str
      } catch {
        case _: IllegalArgumentException =>
      }
    }
    logger.info(msg)
  }
}
