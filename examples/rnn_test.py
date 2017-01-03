# Kyuhong Shim 2016

import numpy as np
import theano
import theano.tensor as T

from theano_lemon.layers.recurrent import ElmanRecurrentLayer

RNN = ElmanRecurrentLayer(2, 3, out_activation = 'relu', output_return_index = [0,1,2,3],
                          precompute=True, unroll=True,
                          gradient_steps = 4,
                          backward=False, name='RNNtest')

W_value = np.array([[-1, 0, 1], [0, 1, 1]], dtype = theano.config.floatX)
U_value = np.array([[-2, -1, 0], [1, 2, 3], [0, 0, 1]], dtype = theano.config.floatX)

RNN.W.set_value(W_value)
RNN.U.set_value(U_value)

print('W', RNN.W.get_value())
print('U', RNN.U.get_value())

input = np.array([[[0, 1], [1, 2], [2, 3], [-1, -2]],
                  [[-1, 2], [0, 0], [1, 2], [-1, -1]],
                  [[0, 1], [2, 1], [-1,-1], [0,0]]], dtype = theano.config.floatX)
mask = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1]], dtype = 'int32')
hidden_init = np.array([[1,1,1], [2,2,2], [-1,-1,-1]], dtype = theano.config.floatX)

X = T.ftensor3('X')
M = T.imatrix('M')
HI = T.fmatrix('HI')

print('input shape', input.shape, input)
print('mask shape', mask.shape, mask)
print('hidden init shape', hidden_init.shape, hidden_init)

result, params, updates = RNN.get_output(X, M, HI)

func = theano.function(inputs = [X,M,HI], outputs = [result], 
                       allow_input_downcast = True)

func_result = func(input, mask, hidden_init)[0]

print('Result', func_result.shape)
print(func_result)

print('Done!')