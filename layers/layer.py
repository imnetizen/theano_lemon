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

class BaseRecurrentLayer(BaseLayer):
    def __init__(self, gradient_steps = -1, state_save_index = -1, unroll = True, name = None):
        self.gradient_steps = gradient_steps
        self.state_save_index = state_save_index
        self.unroll = unroll
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

