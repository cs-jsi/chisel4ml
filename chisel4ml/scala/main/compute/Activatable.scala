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
package chisel4ml.compute

import chisel3._
import chisel4ml.HasLayerWrap
import chisel4ml.util._
import dsptools.numbers.implicits._
import lbir.Activation.{BINARY_SIGN, NO_ACTIVATION, RELU}

// ACTIVATION FUNCTIONS
trait Activatable[A <: Data, O <: Data] {
  def actFn: (A, A) => O
}

object ActivatableImplementations {
  trait ActivatableSIntUInt extends Activatable[SInt, UInt] with HasLayerWrap {
    override def actFn = cfg.activation match {
      case RELU => reluFn(cfg.output.dtype.bitwidth)
      case _    => throw new RuntimeException
    }
  }
  trait ActivatableSIntSInt extends Activatable[SInt, SInt] with HasLayerWrap {
    override def actFn = cfg.activation match {
      case NO_ACTIVATION => linearFn(cfg.output.dtype.bitwidth)
      case _             => throw new RuntimeException
    }
  }
  trait ActivatableSIntBool extends Activatable[SInt, Bool] with HasLayerWrap {
    override def actFn = cfg.activation match {
      case BINARY_SIGN => signFn[SInt]
      case _           => throw new RuntimeException
    }
  }
  trait ActivatableUIntBool extends Activatable[UInt, Bool] with HasLayerWrap {
    override def actFn = cfg.activation match {
      case BINARY_SIGN => signFn[UInt]
      case _           => throw new RuntimeException
    }
  }
  trait ActivatableUIntUInt extends Activatable[UInt, UInt] with HasLayerWrap {
    override def actFn = cfg.activation match {
      case _ => throw new RuntimeException
    }
  }
  trait ActivatableUIntSInt extends Activatable[UInt, SInt] with HasLayerWrap {
    override def actFn = cfg.activation match {
      case _ => throw new RuntimeException
    }
  }

  def signFn[A <: Bits with Num[A]] = (act: A, thresh: A) => act >= thresh
  def reluNoSaturation(act: SInt, thresh: SInt): UInt = Mux((act - thresh) > 0.S, (act - thresh).asUInt, 0.U)
  def reluFn(bitwidth:      Int) = (act: SInt, thresh: SInt) => saturateFnU(reluNoSaturation(act, thresh), bitwidth)
  def linearFn(bitwidth:    Int) = (act: SInt, thresh: SInt) => saturateFnS(act - thresh, bitwidth)
}
