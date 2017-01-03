# Kyuhong Shim 2016

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from .layer import BaseRecurrentLayer


class ElmanRecurrentLayer(BaseRecurrentLayer):

    def __init__(self, input_shape, output_shape,
                 out_activation = 'tanh',
                 gradient_steps=-1,
                 output_return_index=[-1],  # should be list
                 precompute = False, unroll=False, backward=False, name=None):
        super(ElmanRecurrentLayer, self).__init__(gradient_steps, output_return_index,
                                                  precompute, unroll, backward, name)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.out_activation = out_activation

        W = np.zeros((input_shape, output_shape)).astype(theano.config.floatX)
        self.W = theano.shared(W, self.name + '_weight_W')
        self.W.tag = 'weight'
        U = np.zeros((output_shape, output_shape)).astype(theano.config.floatX)
        self.U = theano.shared(U, self.name + '_weight_U')
        self.U.tag = 'weight'
        b = np.zeros((output_shape,)).astype(theano.config.floatX)
        self.b = theano.shared(b, self.name + '_bias')
        self.b.tag = 'bias'

    def _compute_output(self, inputs, masks, hidden_init):
        # inputs are (n_batch, n_timesteps, n_features)
        # change to (n_timesteps, n_batch, n_features)
        inputs = inputs.dimshuffle(1, 0, *range(2, inputs.ndim))
        # masks are (n_batch, n_timesteps)
        masks = masks.dimshuffle(1, 0, 'x')
        sequence_length = inputs.shape[0]
        batch_num = inputs.shape[1]

        if self.out_activation is 'tanh':
            activation = T.tanh
        elif self.out_activation is 'sigmoid':
            activation = T.nnet.sigmoid
        elif self.out_activation is 'relu':
            activation = T.nnet.relu
        else:
            raise ValueError('Not yet considered')

        if self.precompute:
            additional_dims = tuple(inputs.shape[k] for k in range(2, inputs.ndim))
            inputs = T.reshape(inputs, (sequence_length*batch_num,) + additional_dims)
            inputs = T.dot(inputs, self.W)
            additional_dims = tuple(inputs.shape[k] for k in range(1, inputs.ndim))
            inputs = T.reshape(inputs, (sequence_length, batch_num,) + additional_dims)

        def step(input, hidden):
            if self.precompute:
                return activation(input + T.dot(hidden, self.U) + self.b)
            else:
                return activation(T.dot(input, self.W) + T.dot(hidden, self.U) + self.b)

        def step_masked(input, mask, hidden):
            hidden_computed = step(input, hidden)
            return T.switch(mask, hidden_computed, hidden)

        if self.unroll:
            assert self.gradient_steps > 0
            counter = range(self.gradient_steps)
            if self.backward:
                counter = counter[::-1]
            iter_output = []
            outputs_info = [hidden_init]
            for index in counter:
                step_input = [inputs[index], masks[index]] + outputs_info
                step_output = step_masked(*step_input)
                iter_output.append(step_output)
                outputs_info = [iter_output[-1]]
            hidden_output = T.stack(iter_output, axis=0)

        else:
            hidden_output = theano.scan(fn=step_masked,
                                        sequences=[inputs, masks],
                                        outputs_info=[hidden_init],
                                        # non_sequences=[self.W, self.U, self.b],
                                        go_backwards=self.backward,
                                        n_steps = None,
                                        truncate_gradient=self.gradient_steps)[0]

        hidden_output_return = hidden_output[self.output_return_index]
        # change to (n_batch, n_timesteps, n_features)
        hidden_output_return = hidden_output_return.dimshuffle(1, 0, *range(2, hidden_output_return.ndim))
        #hidden_output_return = hidden_output
        if self.backward:
            hidden_output_return = hidden_output_return[:, ::-1]

        return hidden_output_return

    def _collect_params(self):
        return [self.W, self.U, self.b]

    def _collect_updates(self):
        return OrderedDict()
