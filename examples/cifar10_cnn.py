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
from theano_lemon.data.generators import BaseGenerator, ImageGenerator
from theano_lemon.data.cifar10 import load_cifar10

from theano_lemon.controls.history import HistoryWithEarlyStopping
from theano_lemon.controls.scheduler import LearningRateMultiplyScheduler
from theano_lemon.layers.dense import DenseLayer
from theano_lemon.layers.activation import ReLU, Softmax
from theano_lemon.layers.convolution import Convolution2DLayer, Padding2DLayer
from theano_lemon.layers.pool import Pooling2DLayer
from theano_lemon.layers.shape import Flatten3DLayer
from theano_lemon.layers.dropout import DropoutLayer
from theano_lemon.layers.normalization import BatchNormalization1DLayer, BatchNormalization2DLayer
from theano_lemon.misc import merge_dicts, split_data, get_inputs

np.random.seed(99999)
base_datapath = 'C:/Users/skhu2/Dropbox/Project/data/'
#base_datapath = '/home/khshim/data/'

def train(name = 'cifar10'):

    train_data, train_label, test_data, test_label = load_cifar10(base_datapath, 'tensor')
    train_data, train_label, valid_data, valid_label = split_data(train_data, train_label, 45000)

    train_gen = ImageGenerator('train', 64)
    train_gen.initialize(train_data, train_label)
    train_gen.rgb_to_yuv()
    gcn_mean, gcn_std = train_gen.gcn()
    #pc_matrix = train_gen.zca()
    train_gen.set_flip_lr_true()
    test_gen = ImageGenerator('test', 64)
    test_gen.initialize(test_data, test_label)
    test_gen.rgb_to_yuv()
    test_gen.gcn(gcn_mean, gcn_std)
    #test_gen.zca(pc_matrix)
    test_gen.set_flip_lr_true()
    valid_gen = ImageGenerator('valid', 64)
    valid_gen.initialize(valid_data, valid_label)
    valid_gen.rgb_to_yuv()
    valid_gen.gcn(gcn_mean, gcn_std)
    #valid_gen.zca(pc_matrix)
    valid_gen.set_flip_lr_true()

    x= T.ftensor4('X')
    y = T.ivector('y')

    # network like VGG-16

    graph = BaseGraph(name = name)
    graph.set_input(x)
    graph.add_layers([Padding2DLayer((3,32,32), (3,34,34), (1,1,1,1), name = 'p1'),
                      Convolution2DLayer((3,34,34), (64,32,32), (3,3), use_bias = False, name = 'c1'),
                      BatchNormalization2DLayer((64,32,32), name = 'bn1'),
                      ReLU(name = 'r1'),
                      
                      DropoutLayer(0.3, rescale = True, name = 'drop1'),

                      Padding2DLayer((64,32,32), (64,34,34), (1,1,1,1), name = 'p2'),
                      Convolution2DLayer((64,34,34), (64,32,32), (3,3), use_bias = False, name = 'c2'),
                      BatchNormalization2DLayer((64,32,32), name = 'bn2'),
                      ReLU(name = 'r2'),                      

                      Pooling2DLayer((64,32,32), (64,16,16), (2,2), name = 'pool1'),

                      Padding2DLayer((64,16,16), (64,18,18), (1,1,1,1), name = 'p3'),
                      Convolution2DLayer((64,18,18), (128,16,16), (3,3), use_bias = False, name = 'c3'),
                      BatchNormalization2DLayer((128,16,16), name = 'bn3'),
                      ReLU(name = 'r3'),                      
                      DropoutLayer(0.4, rescale = True, name = 'drop2'),

                      Padding2DLayer((128,16,16), (128,18,18), (1,1,1,1), name = 'p4'),
                      Convolution2DLayer((128,18,18), (128,16,16), (3,3), use_bias = False, name = 'c4'),
                      BatchNormalization2DLayer((128,16,16), name = 'bn4'),
                      ReLU(name = 'r4'),                      

                      Pooling2DLayer((128,16,16), (128,8,8), (2,2), name = 'pool2'),

                      Padding2DLayer((128,8,8), (128,10,10), (1,1,1,1), name = 'p5'),
                      Convolution2DLayer((128,10,10), (256,8,8), (3,3), use_bias = False, name = 'c5'),
                      BatchNormalization2DLayer((256,8,8), name = 'bn5'),
                      ReLU(name = 'r5'),                      
                      DropoutLayer(0.4, rescale = True, name = 'drop3'),

                      Padding2DLayer((256,8,8), (256,10,10), (1,1,1,1), name = 'p6'),
                      Convolution2DLayer((256,10,10), (256,8,8), (3,3), use_bias = False, name = 'c6'),
                      BatchNormalization2DLayer((256,8,8), name = 'bn6'),
                      ReLU(name = 'r6'),                      
                      DropoutLayer(0.4, rescale = True, name = 'drop4'),

                      Padding2DLayer((256,8,8), (256,10,10), (1,1,1,1), name = 'p7'),
                      Convolution2DLayer((256,10,10), (256,8,8), (3,3), use_bias = False, name = 'c7'),
                      BatchNormalization2DLayer((256,8,8), name = 'bn7'),
                      ReLU(name = 'r7'),                      
                      
                      Pooling2DLayer((256,8,8), (256,4,4), (2,2), name = 'pool3'),

                      Padding2DLayer((256,4,4), (256,6,6), (1,1,1,1), name = 'p8'),
                      Convolution2DLayer((256,6,6), (512,4,4), (3,3), use_bias = False, name = 'c8'),
                      BatchNormalization2DLayer((512,4,4), name = 'bn8'),
                      ReLU(name = 'r8'),                      
                      DropoutLayer(0.4, rescale = True, name = 'drop5'),

                      Padding2DLayer((512,4,4), (512,6,6), (1,1,1,1), name = 'p9'),
                      Convolution2DLayer((512,6,6), (512,4,4), (3,3), use_bias = False, name = 'c9'),
                      BatchNormalization2DLayer((512,4,4), name = 'bn9'),
                      ReLU(name = 'r9'),                      
                      DropoutLayer(0.4, rescale = True, name = 'drop6'),

                      Padding2DLayer((512,4,4), (512,6,6), (1,1,1,1), name = 'p10'),
                      Convolution2DLayer((512,6,6), (512,4,4), (3,3), use_bias = False, name = 'c10'),
                      BatchNormalization2DLayer((512,4,4), name = 'bn10'),
                      ReLU(name = 'r10'),                      

                      Pooling2DLayer((512,4,4), (512,2,2), (2,2), name = 'pool4'),

                      Padding2DLayer((512,2,2), (512,4,4), (1,1,1,1), name = 'p11'),
                      Convolution2DLayer((512,4,4), (512,2,2), (3,3), use_bias = False, name = 'c11'),
                      BatchNormalization2DLayer((512,2,2), name = 'bn11'),
                      ReLU(name = 'r11'),
                      DropoutLayer(0.4, rescale = True, name = 'drop7'),

                      Padding2DLayer((512,2,2), (512,4,4), (1,1,1,1), name = 'p12'),
                      Convolution2DLayer((512,4,4), (512,2,2), (3,3), use_bias = False, name = 'c12'),
                      BatchNormalization2DLayer((512,2,2), name = 'bn12'),
                      ReLU(name = 'r12'),
                      DropoutLayer(0.4, rescale = True, name = 'drop8'),

                      Padding2DLayer((512,2,2), (512,4,4), (1,1,1,1), name = 'p13'),
                      Convolution2DLayer((512,4,4), (512,2,2), (3,3), use_bias = False, name = 'c13'),
                      BatchNormalization2DLayer((512,2,2), name = 'bn13'),
                      ReLU(name = 'r13'),
                      
                      Pooling2DLayer((512,2,2), (512,1,1), (2,2), name = 'pool5'),
                     
                      Flatten3DLayer((512,1,1), 512, name = 'flatten1'),
                      DropoutLayer(0.5, rescale = True, name = 'drop9'),

                      DenseLayer(512, 1024, use_bias = False, name = 'd1'),
                      BatchNormalization1DLayer(1024, name = 'bn14'),
                      ReLU(name = 'r14'),
                      DropoutLayer(0.5, rescale = True, name = 'drop10'),

                      DenseLayer(1024, 1024, use_bias = False, name = 'd2'),
                      BatchNormalization1DLayer(1024, name = 'bn15'),
                      ReLU(name = 'r15'),
                      DropoutLayer(0.5, rescale = True, name = 'drop11'),

                      DenseLayer(1024, 10, name = 'd3'),
                      Softmax(name = 'softmax1')
                      ])

    output = graph.get_output()
    layer_params = graph.get_params()
    internal_updates = graph.get_updates()

    loss = CategoricalCrossentropy().get_loss(output, y) 
    accuracy = CategoricalAccuracy().get_loss(output, y)

    adam = Adam()
    external_updates = adam.get_update(loss, layer_params)
    optimzer_params = adam.get_internals()
    params = layer_params + optimzer_params
    inputs = get_inputs(loss)

    params_saver = BaseParameter(params, name+'_params/')    

    GlorotNormal().initialize(params_saver.filter_params('weight'))
    # Constant(0).initialize(params_saver.filter_params('bias'))
    params_saver.save_params()

    train_func = theano.function(inputs,
                                 [loss, accuracy],
                                 updates = merge_dicts(external_updates, internal_updates),
                                 allow_input_downcast = True)
    test_func = theano.function(inputs,
                                 [loss, accuracy],
                                 allow_input_downcast = True)

    lr_scheduler = LearningRateMultiplyScheduler(adam.lr, 0.2)
    hist = HistoryWithEarlyStopping(50, 5)

    train_start_time = time.clock()

    change_lr = False
    stop_run = False
    for epoch in range(1000):
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
        elif checker == 3:
            change_lr = False
            stop_run = False
        else:
            raise NotImplementedError('Not supported checker type')

    train_end_time = time.clock()
    print('...Total Train time:', train_end_time - train_start_time)

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
    train('cifar10_trying_under10')