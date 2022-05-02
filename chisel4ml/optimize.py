from chisel4ml.optimizations import qkeras_opt_list

import tensorflow as tf

import copy

MAX_PASSES = 3


def qkeras_model(model):
    "Applys optimization passes to the model, and returns a dummy model that can be transformed into a LBIR model."
    layers = copy.deepcopy(model.layers)  # layers in keras are read-only

    def _rolling_window(list, degree):
        for i in range(len(list)-degree+1):
            yield [list[i+o] for o in range(degree)]

    # Some layers are wrapped in other layers (pruning layer i.e.) in the first pass we unwrapp it and then
    # we apply other optimizations.
    for _ in range(MAX_PASSES):
        for opt in qkeras_opt_list:
            for lslice in _rolling_window(layers, opt.num_layers):
                assert len(lslice) > 0
                if opt.is_applicable(lslice):
                    lslice = opt(lslice)  # TODO: is this safe?

    # Re-create the model
    new_model = tf.keras.models.Sequential()
    for layer in layers:
        new_model.add(layer)

    return new_model
