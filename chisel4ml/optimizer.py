from chisel4ml.optimizations import qkeras_opt_factory

import tensorflow as tf

import copy

MAX_PASSES = 3


def optimize_model(model):
    "Applys optimization passes to the model, and returns a dummy model that can be transformed into a LBIR model."
    layers = copy.deepcopy(model.layers)  # layers in keras are read-only

    # Some layers are wrapped in other layers (pruning layer i.e.) in the first pass we unwrapp it and then
    # we apply other optimizations.
    for _ in range(MAX_PASSES):
        for i, layer in enumerate(list(layers)):
            opt = qkeras_opt_factory(layer)
            layers[i:i+opt.num_layers] = opt(layers[i:i+opt.num_layers])

    # Re-create the model
    new_model = tf.keras.models.Sequential()
    for layer in layers:
        new_model.add(layer)

    return new_model
