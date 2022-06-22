from chisel4ml.optimizations import qkeras_opt_list

import tensorflow as tf

import copy
import collections
import itertools
import logging
log = logging.getLogger(__name__)


def qkeras_model(model):
    "Applys optimization passes to the model, and returns a dummy model that can be transformed into a LBIR model."
    layers = copy.deepcopy(model.layers)  # layers in keras are read-only

    def sliding_window(iterable, size):
        iterable = iter(iterable)
        window = collections.deque(
            itertools.islice(iterable, size-1),
            maxlen=size
        )

        for item in enumerate(iterable):
            window.append(item[1])
            yield window, item[1]

    # Some layers are wrapped in other layers (pruning layer i.e.) in the first pass we unwrapp it and then
    # we apply other optimizations.

    for opt in qkeras_opt_list:
        nlayers = []
        for lslice, item in sliding_window(layers, opt.num_layers):
            if any(hasattr(x, 'c4ml_remove_layer') for x in lslice):
                continue
            if opt.is_applicable(lslice):
                opt(lslice)
        for lay in layers:
            if not hasattr(lay, 'c4ml_remove_layer'):
                if hasattr(lay, 'SKIP'):
                    delattr(lay, 'SKIP')
                nlayers.append(lay)

        layers = nlayers

    return tf.keras.models.Sequential(layers)
