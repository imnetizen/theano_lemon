# Kyuhong Shim 2016

import numpy as np
import os
from collections import OrderedDict

class BaseParameter(object):
    def __init__(self, params, paramdir):
        self.params = params
        self.paramdir = paramdir
        if not os.path.exists(self.paramdir):
            os.makedirs(self.paramdir)        
    def filter_params(self, tag):
        return [pp for pp in self.params if pp.tag == tag]
    def save_params(self, postfix = None):
        print('...weight save done')
        for pp in self.params:
            if postfix is None:
                np.save(self.paramdir + pp.name + '.npy', pp.get_value())
            else:
                np.save(self.paramdir + pp.name + '_' + tag + '.npy', pp.get_value())
    def load_params(self, postfix = None):
        print('...weight load done')
        for pp in self.params:
            if postfix is None:
                pp.set_value(np.load(self.paramdir + pp.name + '.npy'))
            else:
                pp.set_value(np.load(self.paramdir + pp.name + '_' + postfix + '.npy'))
 
            