# Kyuhong Shim 2016
"""
Objectives set loss function for graph.
Return a loss.
"""

import numpy as np
import theano
import theano.tensor as T


class BaseObjective(object):

    def __init__(self, name=None):
        self.name = name

    def get_loss(self, predict, label):
        raise NotImplementedError('Abstract class method')


class CategoricalCrossentropy(BaseObjective):

    def __init__(self, name=None, stabilize=False):
        self.tag = 'loss'
        self.stabilize = stabilize
        super(CategoricalCrossentropy, self).__init__(name)

    def get_loss(self, predict, label):
        if self.stabilize:
            return T.mean(T.nnet.categorical_crossentropy(T.clip(predict, 1e-6, 1.0 - 1e-6), label))
        else:
            return T.mean(T.nnet.categorical_crossentropy(predict, label))


class CategoricalAccuracy(BaseObjective):

    def __init__(self, name=None):
        self.tag = 'accuracy'
        super(CategoricalAccuracy, self).__init__(name)

    def get_loss(self, predict, label):
        return T.mean(T.eq(T.argmax(predict, axis=-1), label))


class SquareError(BaseObjective):

    def __init__(self, name=None):
        self.tag = 'loss'
        super(SquareError, self).__init__(name)

    def get_loss(self, predict, label):
        return T.mean(T.square(predict - label))
