# Kyuhong Shim 2016

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

class BaseLayer(object):
    def __init__(self, name = None):
        self.name = name
    def _compute_output(self, inputs):
        raise NotImplementedError('Abstract class method')
    def _collect_params(self): # Trained by optimizers
        raise NotImplementedError('Abstarct class method')
    def _collect_updates(self): # Additional updates
        raise NotImplementedError('Abstract class method')
    def get_output(self, inputs):
        output = self._compute_output(inputs)
        params = self._collect_params()
        updates = self._collect_updates()
        return output, params, updates

