# Kyuhong Shim 2016

import numpy as np

class BaseGenerator(object):
    def __init__(self, name = None, batchsize = 128):
        self.name = name
        self.batchsize = batchsize
    def initialize(self, data, label):
        self.data = data
        self.label = label
        self.ndata = data.shape[0]
        self.order = np.random.permutation(self.ndata)
        self.max_index = self.ndata // self.batchsize
    def shuffle(self):
        self.order = np.random.permutation(self.ndata)
    def change_batchsize(self, newbatchsize):
        self.batchsize = newbatchsize
        self.max_index = self.ndata // self.batchsize
    def get_minibatch(self, index):
        assert index <= self.max_index
        return (self.data[self.order[self.batchsize *index: self.batchsize * (index+1)]],
                self.label[self.order[self.batchsize *index: self.batchsize * (index+1)]])
    def get_fullbatch(self):
        return (self.data, self.label)