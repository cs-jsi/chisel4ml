from chisel4ml.optimizations import qkeras_opt_list

import tensorflow as tf

import copy
import collections
import itertools
import os
import logging

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def qkeras_model(model):
    "Applys optimization passes to the model, and returns a dummy model that can be transformed into a LBIR model."
    layers = copy.deepcopy(model.layers)  # layers in keras are read-only

    def sliding_window(iterable, size):
        iterable = iter(iterable)
        window = collections.deque(
            itertools.islice(iterable, size-1),
            maxlen=size
        )

        for item in iterable:
            window.append(item)
            yield window

    # Some layers are wrapped in other layers (pruning layer i.e.) in the first pass we unwrapp it and then
    # we apply other optimizations.
    for opt in qkeras_opt_list:
        for lslice in sliding_window(layers, opt.num_layers):
            if opt.is_applicable(lslice):
                lslice = opt(lslice)

        layers = list(filter(lambda l: not hasattr(l, 'c4ml_remove_layer'), layers))

    return tf.keras.models.Sequential(layers)
