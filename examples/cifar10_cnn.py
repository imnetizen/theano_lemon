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
from theano_lemon.data.cifar10 import load_cifar10

from theano_lemon.controlls.history import HistoryWithEarlyStopping
from theano_lemon.controlls.scheduler import LearningRateMultiplyScheduler
from theano_lemon.layers.dense import DenseLayer
from theano_lemon.layers.activation import ReLU, Softmax
from theano_lemon.layers.convolution import Convolution2DLayer, Padding2DLayer
from theano_lemon.layers.pool import Pooling2DLayer
from theano_lemon.layers.shape import Flatten3DLayer
from theano_lemon.layers.normalization import BatchNormalization1DLayer, BatchNormalization2DLayer
from theano_lemon.misc import merge_dicts, split_data, get_inputs

np.random.seed(99999)
#base_datapath = 'D:/Dropbox/Project/data/'
base_datapath = 'C:/Users/skhu2/Dropbox/Project/data/'

def train(name = 'cifar10'):

    train_data, train_label, test_data, test_label = load_cifar10(base_datapath, 'tensor')
    train_data, train_label, valid_data, valid_label = split_data(train_data, train_label, 45000)

    train_gen = BaseGenerator('train', 64)
    train_gen.initialize(train_data, train_label)
    test_gen = BaseGenerator('test', 64)
    test_gen.initialize(test_data, test_label)
    valid_gen = BaseGenerator('valid', 64)
    valid_gen.initialize(valid_data, valid_label)

    x= T.ftensor4('X')
    y = T.ivector('y')

    graph = BaseGraph(name = name)
    graph.set_input(x)
    graph.add_layers([Convolution2DLayer((3,32,32), (64,30,30), (3,3), use_bias = False, name = 'c1'),
                      BatchNormalization2DLayer((64,30,30), name = 'bn1'),
                      ReLU(name = 'r1'),
                      Padding2DLayer((64,30,30), (64,32,32), (1,1,1,1), name = 'p1'),

                      Convolution2DLayer((64,32,32), (64,30,30), (3,3), use_bias = False, name = 'c2'),
                      BatchNormalization2DLayer((64,30,30), name = 'bn2'),
                      ReLU(name = 'r2'),
                      Padding2DLayer((64,30,30), (64,32,32), (1,1,1,1), name = 'p2'),

                      Pooling2DLayer((64,32,32), (64,16,16), (2,2), name = 'pool1'),

                      Convolution2DLayer((64,16,16), (64,14,14), (3,3), use_bias = False, name = 'c3'),
                      BatchNormalization2DLayer((64,14,14), name = 'bn3'),
                      ReLU(name = 'r3'),
                      Padding2DLayer((64,14,14), (64,16,16), (1,1,1,1), name = 'p3'),

                      Convolution2DLayer((64,16,16), (64,14,14), (3,3), use_bias = False, name = 'c4'),
                      BatchNormalization2DLayer((64,14,14), name = 'bn4'),
                      ReLU(name = 'r4'),
                      Padding2DLayer((64,14,14), (64,16,16), (1,1,1,1), name = 'p4'),

                      Pooling2DLayer((64,16,16), (64,8,8), (2,2), name = 'pool2'),

                      Convolution2DLayer((64,8,8), (64,6,6), (3,3), use_bias = False, name = 'c5'),
                      BatchNormalization2DLayer((64,6,6), name = 'bn5'),
                      ReLU(name = 'r5'),
                      Padding2DLayer((64,6,6), (64,8,8), (1,1,1,1), name = 'p5'),

                      Convolution2DLayer((64,8,8), (64,6,6), (3,3), use_bias = False, name = 'c6'),
                      BatchNormalization2DLayer((64,6,6), name = 'bn6'),
                      ReLU(name = 'r6'),
                      
                      Flatten3DLayer((64,6,6), 2304, name = 'flatten1'),

                      DenseLayer(2304, 2048, use_bias = False, name = 'd1'),
                      BatchNormalization1DLayer(2048, name = 'bn7'),
                      ReLU(name = 'r7'),
                      DenseLayer(2048, 2048, use_bias = False, name = 'd2'),
                      BatchNormalization1DLayer(2048, name = 'bn8'),
                      ReLU(name = 'r8'),
                      DenseLayer(2048, 10, name = 'd3'),
                      Softmax(name = 'softmax1')
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
    hist = HistoryWithEarlyStopping(3, 5)

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

    params_saver.print_param_statistics(['weight', 'bias'])

    return hist

if __name__ == '__main__':
    train('cifar10_cnn')