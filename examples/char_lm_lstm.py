# Kyuhong Shim 2016

import time
import numpy as np
#import theano
#import theano.tensor as T

#from theano_lemon.initializers import GlorotNormal
#from theano_lemon.optimizers import Adam
#from theano_lemon.objectives import CategoricalAccuracy, CategoricalCrossentropy
#from theano_lemon.graph import BaseGraph
#from theano_lemon.parameters import BaseParameter
from theano_lemon.data.generators import CharacterGenerator
from theano_lemon.data.nietzsche import load_nietzsche
#from theano_lemon.misc import split_data, get_inputs, merge_dicts

#from theano_lemon.controls.history import HistoryWithEarlyStopping
#from theano_lemon.controls.scheduler import LearningRateMultiplyScheduler
#from theano_lemon.layers.dense import DenseLayer
#from theano_lemon.layers.activation import ReLU, Softmax
#from theano_lemon.layers.recurrent import LSTMRecurrentLayer

np.random.seed(99999)
#base_datapath = 'D:/Dropbox/Project/data/'
base_datapath = 'C:/Users/skhu2/Dropbox/Project/data/'
#base_datapath = '/home/khshim/data/'

def train(name = 'char_lm_lstm'):

    nietzsche_sentences = load_nietzsche(base_datapath)
    data_gen = CharacterGenerator('data', batchsize=32, bucket=5)
    data_gen.initialize(nietzsche_sentences, sort=True)
    for i in range(data_gen.ndata):
        print(len(data_gen.data[i]))
    print(data_gen.bucket_key)

if __name__ == '__main__':
    train('char_lm_lstm')