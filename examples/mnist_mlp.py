# Kyuhong Shim 2016

import time
import numpy as np
import theano
import theano.tensor as T

from theano_lemon.initializers import GlorotNormal, Constant
from theano_lemon.optimizers import Adam
from theano_lemon.objectives import CategoricalAccuracy, CategoricalCrossentropy
from theano_lemon.graph import BaseGraph
from theano_lemon.parameters import BaseParameter
from theano_lemon.data.generators import BaseGenerator
from theano_lemon.data.mnist import load_mnist
from theano_lemon.misc import split_data, get_inputs, merge_dicts

from theano_lemon.controlls.history import HistoryWithEarlyStopping
from theano_lemon.controlls.scheduler import LearningRateMultiplyScheduler
from theano_lemon.layers.dense import DenseLayer
from theano_lemon.layers.activation import ReLU, Softmax
from theano_lemon.layers.normalization import BatchNormalization1DLayer
from theano_lemon.layers.dropout import DropoutLayer

np.random.seed(99999)
base_datapath = 'D:/Dropbox/Project/data/'
#base_datapath = 'C:/Users/skhu2/Dropbox/Project/data/'
#base_datapath = '/home/khshim/data/'

def train(name = 'mnist'):

    train_data, train_label, test_data, test_label = load_mnist(base_datapath, 'flat')
    train_data, train_label, valid_data, valid_label = split_data(train_data, train_label, 50000)

    train_gen = BaseGenerator('train', 250)
    train_gen.initialize(train_data, train_label)
    test_gen = BaseGenerator('test', 250)
    test_gen.initialize(test_data, test_label)
    valid_gen = BaseGenerator('valid', 250)
    valid_gen.initialize(valid_data, valid_label)

    x= T.fmatrix('X')
    y = T.ivector('y')

    graph = BaseGraph(name = name)
    graph.set_input(x)
    graph.add_layers([DenseLayer(784, 1024, name = 'd1'),
                      ReLU(name = 'r1'),
                      DropoutLayer(name = 'drop1'),
                      DenseLayer(1024, 1024, name = 'd2'),
                      ReLU(name = 'r2'),
                      DenseLayer(1024, 1024, name = 'd3'),
                      ReLU(name = 'r3'),
                      DenseLayer(1024, 1024, name = 'd4'),
                      ReLU(name = 'r4'),
                      DenseLayer(1024, 10, name = 'd5'),
                      Softmax(name = 's1')
                      ])

    output = graph.get_output()
    params = graph.get_params()
    internal_updates = graph.get_updates()

    loss = CategoricalCrossentropy().get_loss(output, y) 
    accuracy = CategoricalAccuracy().get_loss(output, y)

    adam = Adam()
    external_updates = adam.get_update(loss, params)
    opt_internals = adam.get_internals()
    params = params + opt_internals
    inputs = get_inputs(loss)

    params_saver = BaseParameter(params, name+'_params/')    

    GlorotNormal().initialize(params_saver.filter_params('weight'))
    Constant(0).initialize(params_saver.filter_params('bias'))
    params_saver.save_params()

    train_func = theano.function(inputs,
                                 [loss, accuracy],
                                 updates = merge_dicts(external_updates, internal_updates),
                                 allow_input_downcast = True)
    test_func = theano.function(inputs,
                                 [loss, accuracy],
                                 allow_input_downcast = True)

    lr_scheduler = LearningRateMultiplyScheduler(adam.lr, 0.5)
    hist = HistoryWithEarlyStopping(5, 7)

    change_lr = False
    stop_run = False
    for epoch in range(500):
        if stop_run == True:
            params_saver.load_params()
            current_best_loss, current_best_epoch = hist.best_valid_loss()            
            hist.remove_history_after(current_best_epoch)
            break
        if change_lr == True:
            params_saver.load_params()
            lr_scheduler.change_learningrate(epoch)
            current_best_loss, current_best_epoch = hist.best_valid_loss()
            hist.remove_history_after(current_best_epoch)
        train_gen.shuffle()

        print('...Epoch', epoch)
        start_time = time.clock()
        graph.change_flag(1)
        train_loss = []
        train_accuracy = []
        for index in range(train_gen.max_index):
            trainset = train_gen.get_minibatch(index)
            train_batch_loss, train_batch_accuracy = train_func(*trainset)
            train_loss.append(train_batch_loss)
            train_accuracy.append(train_batch_accuracy)
        hist.history['train_loss'].append(np.mean(np.asarray(train_loss)))
        hist.history['train_accuracy'].append(np.mean(np.asarray(train_accuracy)))

        graph.change_flag(-1)
        valid_loss = []
        valid_accuracy = []
        for index in range(valid_gen.max_index):
            validset = valid_gen.get_minibatch(index)
            valid_batch_loss, valid_batch_accuracy = test_func(*validset)
            valid_loss.append(valid_batch_loss)
            valid_accuracy.append(valid_batch_accuracy)
        hist.history['valid_loss'].append(np.mean(np.asarray(valid_loss)))
        hist.history['valid_accuracy'].append(np.mean(np.asarray(valid_accuracy)))
        end_time = time.clock()
        print('......time:', end_time - start_time)

        hist.print_history_recent()
        checker = hist.check_earlystopping()
        if checker == 0:
            params_saver.save_params()
            change_lr = False
            stop_run = False
        elif checker == 1:
            change_lr = True
            stop_run = False
        elif checker == 2:
            change_lr = False
            stop_run = True

    graph.change_flag(-1)
    test_loss = []
    test_accuracy = []
    for index in range(test_gen.max_index):
        testset = test_gen.get_minibatch(index)
        test_batch_loss, test_batch_accuracy = test_func(*testset)
        test_loss.append(test_batch_loss)
        test_accuracy.append(test_batch_accuracy)
    hist.history['test_loss'].append(np.mean(np.asarray(test_loss)))
    hist.history['test_accuracy'].append(np.mean(np.asarray(test_accuracy)))
    hist.print_history_recent(['test_loss', 'test_accuracy'])

    #params_saver.print_param_statistics(['weight', 'bias'])

    return hist

if __name__ == '__main__':
    train('mnist_mlp')