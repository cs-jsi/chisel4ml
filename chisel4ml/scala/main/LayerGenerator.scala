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
package chisel4ml

import chisel3._
import _root_.lbir.{Layer}
import _root_.services.LayerOptions
import _root_.chisel4ml.LBIRStream
import _root_.chisel4ml.sequential.{MaxPool2D, ProcessingElementSequentialConv}

object LayerGenerator {
    // TODO: Rewrite the generation procedure to something more sensisble
    def apply(layer: Layer, options: LayerOptions): Module with LBIRStream = {
        if (layer.ltype == Layer.Type.PREPROC) {
            Module(new FFTWrapper(layer, options))
        } else if (layer.ltype == Layer.Type.MAX_POOL) {
            Module(new MaxPool2D(layer, options))
        } else if (layer.ltype == Layer.Type.CONV2D) {
            Module(ProcessingElementSequentialConv(layer, options))
        } else {
            Module(new ProcessingElementWrapSimpleToSequential(layer, options))
        }
    }
}
