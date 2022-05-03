from chisel4ml.optimizations import qkeras_opt_list

import tensorflow as tf

import copy
import collections
import itertools

NUM_ITERATIONS = 3  # Some optimizations can only be applied after certain others optimizations have been applied.


def qkeras_model(model):
    "Applys optimization passes to the model, and returns a dummy model that can be transformed into a LBIR model."
    layers = copy.deepcopy(model.layers)  # layers in keras are read-only

    def sliding_window(iterable, size):
        iterable = iter(iterable)
        window = collections.deque(
            itertools.islice(iterable, size-1),
            maxlen=size
        )

        for index, item in enumerate(iterable):
            window.append(item)
            yield tuple((index, window))

    # Some layers are wrapped in other layers (pruning layer i.e.) in the first pass we unwrapp it and then
    # we apply other optimizations.
    for _ in range(NUM_ITERATIONS):
        for opt in qkeras_opt_list:
            for i, lslice in sliding_window(layers, opt.num_layers):
                if opt.is_applicable(lslice):
                    layers[i:i+opt.num_layers] = opt(lslice)  # TODO: is this safe?

    # Re-create the model
    new_model = tf.keras.models.Sequential()
    for layer in layers:
        new_model.add(layer)

    return new_model
