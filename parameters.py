# Kyuhong Shim 2016

import numpy as np
import os
import matplotlib.pyplot as plt
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
    def print_param_statistics(self, tag = None, postfix = None):
        if tag is not None:
            print(tag + 'statistics')
            for pp in self.params:
                if pp.tag is not tag:
                    continue
                pvalue = pp.get_value()
                pvalue = np.reshape(pvalue, np.prod(pvalue.shape))
                print('...' + pp.name + ' mean:', np.mean(pvalue), ' std:', np.std(pvalue), ' max:', np.max(pvalue), ' min:', np.min(pvalue))
                fig = plt.figure()
                plt.hist(pvalue, 100, alpha = 0.8)
                if postfix is None:
                    plt.savefig(self.paramdir + pp.name + '_hist.jpg')
                else:
                    plt.savefig(self.paramdir + pp.name + '_' + postfix + 'hist.jpg')
                plt.close(fig)
        else:
            print('all parameter statistics')
            for pp in self.params:
                pvalue = pp.get_value()
                pvalue = np.reshape(pvalue, np.prod(pvalue.shape))
                print('...' + pp.name + ' mean:', np.mean(pvalue), ' std:', np.std(pvalue), ' max:', np.max(pvalue), ' min:', np.min(pvalue))
                fig = plt.figure()
                plt.hist(pvalue, 100, alpha = 0.8)
                if postfix is None:
                    plt.savefig(self.paramdir + pp.name + '_hist.jpg')
                else:
                    plt.savefig(self.paramdir + pp.name + '_' + postfix + 'hist.jpg')
                plt.close(fig)