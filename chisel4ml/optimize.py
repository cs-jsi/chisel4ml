from chisel4ml.optimizations import qkeras_opt_list

from tensorflow_model_optimization.python.core.sparsity.keras import prune
import tensorflow as tf
import qkeras

import collections
import itertools
import logging
log = logging.getLogger(__name__)


def _replace_model_layers(smodel, nlayers):
    clayers = []
    for lay in nlayers:
        if not hasattr(lay, 'c4ml_remove_layer'):
            clayers.append(lay)

    for i in range(0, len(smodel.layers)):
        smodel.pop()

    for lay in clayers:
        smodel.add(lay)


def qkeras_model(model, skip_list=[]):
    "Applys optimization passes to the model, and returns a dummy model that can be transformed into a LBIR model."
    # We first strip the model of any pruning layers.
    #striped_model = prune.strip_pruning(model)

    # Now we cloned
    #smodel = tf.keras.models.Sequential(striped_model.layers)  # We convert any functional models to sequential (needed later)
    nmodel = qkeras.utils.clone_model(model)
    smodel = tf.keras.models.Sequential(nmodel.layers)

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
        if opt.__class__.__name__ in skip_list:
            continue
        nlayers = []
        for lslice, item in sliding_window(smodel.layers, opt.num_layers):
            if any(hasattr(x, 'c4ml_remove_layer') for x in lslice):
                continue
            if opt.is_applicable(lslice):
                opt(lslice)
        for lay in smodel.layers:
            if not hasattr(lay, 'c4ml_remove_layer'):
                if hasattr(lay, 'SKIP'):
                    delattr(lay, 'SKIP')
                nlayers.append(lay)

        _replace_model_layers(smodel, nlayers)

    smodel.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    return smodel
