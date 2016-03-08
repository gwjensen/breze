# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import theano.tensor.nnet

#from theano.tensor.nnet import abstract_conv
from theano.tensor.signal import downsample
from theano.sandbox.cuda.dnn import dnn_conv

from breze.arch.component import transfer as _transfer, loss as _loss
from breze.arch.construct.base import Layer
from breze.arch.util import lookup


class AffineNonlinear(Layer):

    @property
    def n_inpt(self):
        return self._n_inpt

    @property
    def n_output(self):
        return self._n_output

    def __init__(self, inpt, n_inpt, n_output, transfer='identity',
                 use_bias=True, declare=None, name=None):
        self.inpt = inpt
        self._n_inpt = n_inpt
        self._n_output = n_output
        self.transfer = transfer
        self.use_bias = use_bias
        super(AffineNonlinear, self).__init__(declare=declare, name=name)

    def _forward(self):
        self.weights = self.declare((self.n_inpt, self.n_output))

        self.output_in = T.dot(self.inpt, self.weights)

        if self.use_bias:
            self.bias = self.declare(self.n_output)
            self.output_in += self.bias

        f = lookup(self.transfer, _transfer)

        self.output = f(self.output_in)


class Split(Layer):

    def __init__(self, inpt, lengths, axis=1, declare=None, name=None):
        self.inpt = inpt
        self.lengths = lengths
        self.axis = axis
        super(Split, self).__init__(declare, name)

    def _forward(self):
        starts = [0] + np.add.accumulate(self.lengths).tolist()
        stops = starts[1:]
        starts = starts[:-1]

        self.outputs = [self.inpt[:, start:stop] for start, stop
                        in zip(starts, stops)]


class Concatenate(Layer):

    def __init__(self, inpts, axis=1, declare=None, name=None):
        self.inpts = inpts
        self.axis = axis
        super(Concatenate, self).__init__(declare, name)

    def _forward(self):
        concatenated = T.concatenate(self.inpts, self.axis)
        self.output = concatenated


class SupervisedLoss(Layer):

    def __init__(self, target, prediction, loss, comp_dim=1, imp_weight=None,
                 declare=None, name=None):
        self.target = target
        self.prediction = prediction
        self.loss_ident = loss

        self.imp_weight = imp_weight
        self.comp_dim = comp_dim

        super(SupervisedLoss, self).__init__(declare, name)

    def _forward(self):
        f_loss = lookup(self.loss_ident, _loss)

        self.coord_wise = f_loss(self.target, self.prediction)

        if self.imp_weight is not None:
            self.coord_wise *= self.imp_weight

        self.sample_wise = self.coord_wise.sum(self.comp_dim)

        self.total = self.sample_wise.mean()

# class Conv2d(Layer):
#
#     def __init__(self, inpt, inpt_height, inpt_width, n_inpt,
#                  filter_height, filter_width,
#                  n_output, transfer='identity',
#                  n_samples=None,
#                  subsample=(1, 1),
#                  padding=(0, 0),
#                  declare=None, name=None):
#         self.inpt = inpt
#         self.inpt_height = inpt_height
#         self.inpt_width = inpt_width
#         self.n_inpt = n_inpt
#
#         self.border_mode = "half"
#
#         self.filter_height = filter_height
#         self.filter_width = filter_width
#
#         self.n_output = n_output
#         self.transfer = transfer
#         self.n_samples = n_samples
#         self.subsample = subsample
#
#         #self.output_height = ((inpt_height - filter_height + 2*padding[0]) /
#         #                      subsample[0] + 1)
#         #self.output_width = ((inpt_width - filter_width + 2*padding[1]) /
#         #                     subsample[1] + 1)
#         #for the sake of things at the moment I'm just going to set these values
#         self.output_height = inpt_height / subsample[0]
#         self.output_width = inpt_width / subsample[1]
#
#         # to use padding:
#         # we should either pad the input before the convolution
#         # and then use "valid" mode (what is done here)
#         # or use "full" mode and then slice the output
#         # if padding[0] > 0:
#         #     self.inpt_height = inpt_height + 2*padding[0]
#         #     self.inpt_width = inpt_width + 2*padding[1]
#         #     inpt_shape = (n_samples, n_inpt, self.inpt_height, self.inpt_width)
#         #     print(inpt_shape)
#         #     self.inpt = T.alloc(0., *inpt_shape)
#         #     self.inpt = T.set_subtensor(self.inpt[:, :, padding[0]:-padding[0], padding[1]:-padding[1]], inpt)
#
#         if not self.output_height > 0:
#             raise ValueError('inpt height smaller than filter height')
#         if not self.output_width > 0:
#             raise ValueError('inpt width smaller than filter width')
#
#         super(Conv2d, self).__init__(declare=declare, name=name)
#
#     def _forward(self):
#         self.weights = self.declare((
#             self.n_output, self.n_inpt,
#             self.filter_height, self.filter_width))
#         self.bias = self.declare((self.n_output,))
#
#         filter_shp = (self.n_inpt, self.n_output, self.filter_height, self.filter_width)
#
#
#         self.output_in = theano.tensor.nnet.conv2d(
#             self.inpt,
#             self.weights,
#             input_shape=(
#                 self.n_samples,
#                 self.n_inpt,
#                 self.inpt_height,
#                 self.inpt_width
#             ),
#             filter_shape=filter_shp,
#             border_mode=self.border_mode,
#             subsample=self.subsample,
#             filterFlip=True)
#
#         f = lookup(self.transfer, _transfer)
#         self.output = f(self.output_in)

class Conv2d(Layer):

    def __init__(self, inpt, inpt_height, inpt_width, n_inpt,
                 filter_height, filter_width,
                 n_output, transfer='identity',
                 n_samples=None,
                 subsample=(1, 1),
                 padding=(0, 0),
                 declare=None, name=None):
        self.inpt = inpt
        self.inpt_height = inpt_height
        self.inpt_width = inpt_width
        self.n_inpt = n_inpt


        self.border_mode = "valid"

        self.filter_height = filter_height
        self.filter_width = filter_width

        self.n_output = n_output
        self.transfer = transfer
        self.n_samples = n_samples
        self.subsample = subsample

        self.output_height = ((inpt_height - filter_height + 2*padding[0]) /
                              subsample[0] + 1)
        self.output_width = ((inpt_width - filter_width + 2*padding[1]) /
                             subsample[1] + 1)

        # to use padding:
        # we should either pad the input before the convolution
        # and then use "valid" mode
        # or use "full" mode and then slice the output (what is done here)
        if padding[0] > 0:
            self.border_mode = "full"
            self.output_in_height = ((inpt_height - filter_height +
                                      (2*filter_height)-1) /
                                     subsample[0] + 1)
            self.output_in_width = ((inpt_width - filter_width +
                                     (2*filter_width)-1) /
                                    subsample[1] + 1)

        if not self.output_height > 0:
            raise ValueError('inpt height smaller than filter height')
        if not self.output_width > 0:
            raise ValueError('inpt width smaller than filter width')

        super(Conv2d, self).__init__(declare=declare, name=name)

    def _forward(self):
        self.weights = self.declare((
            self.n_output, self.n_inpt,
            self.filter_height, self.filter_width))
        self.bias = self.declare((self.n_output,))
        # self.output_in = dnn_conv(self.inpt,
        #                           self.weights,
        #                           border_mode='valid',
        #                           subsample=self.subsample,
        #                           conv_mode='conv')
        self.output_in = conv.conv2d(
           self.inpt,
           self.weights,
           image_shape=(
               self.n_samples,
               self.n_inpt,
               self.inpt_height,
               self.inpt_width
           ),
           filter_shape=(self.n_output,
                         self.n_inpt,
                         self.filter_height,
                         self.filter_width),
           subsample=self.subsample,
           border_mode=self.border_mode,
        )

        if self.border_mode == "full":
            self.output_in = self.output_in[
                :,
                :,
                self.output_in_height/2 - self.output_height/2 :
                self.output_in_height/2 + self.output_height/2,
                self.output_in_width/2 - self.output_width/2 :
                self.output_in_width/2 + self.output_width/2
            ]

        f = lookup(self.transfer, _transfer)
        self.output = f(self.output_in)

# class Upsample2d(Layer):
#
#     def __init__(self, inpt, inpt_height, inpt_width,
#                  upsample_height, upsample_width,
#                  n_output,
#                  padding=None,
#                  transfer='identity',
#                  declare=None, name=None):
#
#         self.inpt = inpt
#         self.inpt_height = inpt_height
#         self.inpt_width = inpt_width
#         self.upsample_height = upsample_height
#         self.upsample_width = upsample_width
#         self.transfer = transfer
#
#         if upsample_height != upsample_height:
#             raise ValueError("upsample height and upsample width are different, not supported yet")
#
#         self.upsample = upsample_height
#
#         if padding is None:
#             self.padding_left = self.padding_right = self.padding_top = self.padding_bottom = 0
#         elif type(padding) == int:
#             self.padding_left = self.padding_right = self.padding_top = self.padding_bottom = padding
#         elif len(padding) == 2:
#             self.padding_left = self.padding_right = padding[0]
#             self.padding_top = self.padding_bottom = padding[1]
#         elif len(padding) == 4:
#             self.padding_left = padding[0]
#             self.padding_top = padding[1]
#             self.padding_right = padding[2]
#             self.padding_bottom = padding[3]
#         else:
#             raise ValueError("padding is not set properly (either None, int, (int, int), (int, int, int, int))")
#
#         self.output_height = inpt_height * upsample_height + self.padding_left + self.padding_right
#         self.output_width = inpt_width * upsample_width + self.padding_top + self.padding_bottom
#
#         print(self.output_height, self.output_width, self.padding_top, self.inpt_height * self.upsample_height, self.padding_left, self.inpt_width * self.upsample_width)
#
#         self.n_output = n_output
#
#         super(Upsample2d, self).__init__(declare=declare, name=name)
#
#     def _forward(self):
#
#         # repeat = T.extra_ops.repeat(
#         #     T.extra_ops.repeat(self.inpt, self.upsample_height, axis=2),
#         #     self.upsample_width, axis=3
#         # )
#         upsamp = self.inpt.repeat(self.upsample_height, axis=2).repeat(self.upsample_width, axis=3)
#
#         # inpt_shape = self.inpt.shape
#         # output_shape = (inpt_shape[0], inpt_shape[1], inpt_shape[2] * self.upsample_height, inpt_shape[3] * self.upsample_width)
#         #
#         # in_dim = inpt_shape[2] * inpt_shape[3]
#         # out_dim = output_shape[2] * output_shape[3]
#         #
#         # print(in_dim * out_dim)
#         #
#         # upsamp_matrix = T.alloc(0., in_dim, out_dim)
#         # rows = T.arange(in_dim)
#         # cols = rows * self.upsample + (rows / inpt_shape[2] * self.upsample * inpt_shape[3])
#         # upsamp_matrix = T.set_subtensor(upsamp_matrix[rows, cols], 1.)
#         #
#         # flat = self.inpt.reshape((inpt_shape[0], inpt_shape[1], inpt_shape[2] * inpt_shape[3]))
#         #
#         # upsamp_flat = T.dot(flat, upsamp_matrix)
#         #
#         # upsamp = upsamp_flat.reshape(output_shape)
#
#         if self.padding_left == 0 and self.padding_top == 0:
#             self.output_in = upsamp
#         else:
#             self.output_in = T.alloc(0., self.inpt.shape[0], self.inpt.shape[1], self.output_height, self.output_width)
#
#             self.output_in = T.set_subtensor(
#                 self.output_in[
#                     :,
#                     :,
#                     self.padding_top:self.padding_top + self.inpt_height * self.upsample_height,
#                     self.padding_left:self.padding_left + self.inpt_width * self.upsample_width
#                 ],
#                 upsamp
#             )
#
#         f = lookup(self.transfer, _transfer)
#         self.output = f(self.output_in)


# class Deconv2d(Layer):
#
#     def __init__(self, inpt, inpt_height, inpt_width, n_inpt,
#                  filter_height, filter_width,
#                  n_output, subsample=(1, 1), padding=(0, 0),
#                  n_samples=None,
#                  transfer='identity',
#                  declare=None, name=None):
#
#         self.inpt = inpt
#         self.inpt_height = inpt_height
#         self.inpt_width = inpt_width
#         self.n_inpt = n_inpt
#         self.n_output = n_output
#
#         self.filter_height = filter_height
#         self.filter_width = filter_width
#
#         self.output_height = filter_height + subsample[0] * (inpt_height - 1) - 2 * padding[0]
#         self.output_width = filter_width + subsample[1] * (inpt_width - 1) - 2 * padding[1]
#
#         self.subsample = subsample
#         self.padding = padding
#
#         self.transfer = transfer
#
#         self.n_samples = n_samples
#
#         super(Deconv2d, self).__init__(declare=declare, name=name)
#
#     def _forward(self):
#
#         self.weights = self.declare((
#             self.n_inpt, self.n_output,
#             self.filter_height, self.filter_width))
#
#         output_shape = (self.n_samples, self.n_output, self.output_height, self.output_width)
#         filter_shp = (self.n_inpt, self.n_output, self.filter_height, self.filter_width)
#
#         # op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
#         #     imshp=output_shape,
#         #     kshp=filter_shape,
#         #     border_mode=self.padding,
#         #     subsample=self.subsample)
#         #
#         # self.output_in = op(self.weights, self.inpt, output_shape[2:])
#         self.bias = self.declare((self.n_output,))
#
#         self.output_in = theano.tensor.nnet.conv2d(
#             self.inpt,
#             self.weights,
#             input_shape=(
#                 self.n_samples,
#                 self.n_inpt,
#                 self.inpt_height,
#                 self.inpt_width
#             ),
#             filter_shape=filter_shp,
#             border_mode="half",
#             subsample=self.subsample,
#             filterFlip=False)
#
#         f = lookup(self.transfer, _transfer)
#         self.output = f(self.output_in)

class Deconv2d(Layer):
    #This function has a few hard coded values specific to MotionNet
    #Don't reverse the weights passed in, the function will do that.
    def __init__(self, inpt, inpt_height, inpt_width, n_inpt,
                 filter_height, filter_width,
                 n_output, weights=None, bias=None, transfer='identity',
                 n_samples=None,
                 subsample=(1, 1),
                 declare=None, name=None):
        self.inpt = inpt
        self.inpt_height = inpt_height
        self.inpt_width = inpt_width
        self.n_inpt = n_inpt

        self.filter_height = filter_height
        self.filter_width = filter_width

        self.n_output = n_output
        self.weights = weights
        self.bias = bias

        self.transfer = transfer
        self.n_samples = n_samples
        self.subsample = subsample

        self.output_height = inpt_height * subsample[0]
        self.output_width = inpt_width * subsample[1]


        #  use "full" mode and then slice the output. This is mandatory with deconvolution.
        self.border_mode = "full"
        self.output_in_height = ((inpt_height - filter_height +
                                  2*(filter_height - 1)) /
                                 subsample[0] + 1)
        self.output_in_width = ((inpt_width - filter_width +
                                 2*(filter_width - 1)) /
                                subsample[1] + 1)

        if not self.output_height > 0:
            raise ValueError('inpt height smaller than filter height')
        if not self.output_width > 0:
            raise ValueError('inpt width smaller than filter width')

        super(Deconv2d, self).__init__(declare=declare, name=name)

    def _forward(self):
        #commented this and the bias out because the model currently isn't using the same filters for both conv and deconv...
        # if self.weights == None:
        #     self.weights = self.declare((
        #         self.n_output, self.n_inpt,
        #         self.filter_height, self.filter_width))
        # else:
        #     #deconvolution is correlation so we reverse the weights. The first two columns as switched to account for going backwards.
        #     #self.weights = self.weights.dimshuffle(1, 0, 2, 3)[:, :, ::-1, ::-1]
        #     self.weights = (self.weights[:, :, ::-1, ::-1])

        self.weights = self.declare((
            self.n_output, self.n_inpt,
            self.filter_height, self.filter_width))

        self.bias = self.declare((self.n_output,))

        # if self.bias == None:
        #     self.bias = self.declare((self.n_output,))

        # self.output_in = dnn_conv(self.inpt,
        #                           self.weights,
        #                           border_mode='half',
        #                           subsample=self.subsample,
        #                           conv_mode='cross')
        self.output_in = conv.conv2d(
            self.inpt,
            self.weights,
            image_shape=(
                self.n_samples,
                self.n_inpt,
                self.inpt_height,
                self.inpt_width
            ),
            filter_shape=(self.n_output,
                          self.n_inpt,
                          self.filter_height,
                          self.filter_width),
            subsample=self.subsample,
            border_mode=self.border_mode,
        )

        self.output_in = self.output_in[
            :,
            :,
            self.output_in_height/2 - self.output_height/2 :
            self.output_in_height/2 + self.output_height/2,
            self.output_in_width/2 - self.output_width/2 :
            self.output_in_width/2 + self.output_width/2
        ]

        f = lookup(self.transfer, _transfer)
        self.output = f(self.output_in)


class MaxPool2d(Layer):

    def __init__(self, inpt, inpt_height, inpt_width, pool_height, pool_width,
                 n_output,
                 transfer='identity',
                 st=None,
                 declare=None, name=None):
        self.inpt = inpt
        self.inpt_height = inpt_height
        self.inpt_width = inpt_width
        self.pool_height = pool_height
        self.pool_width = pool_width
        if st is None:
            st = (pool_height, pool_width)
        self.st = st  # stride
        self.transfer = transfer

        self.output_height = (inpt_height - pool_height) / st[0] + 1
        self.output_width = (inpt_width - pool_width) / st[1] + 1

        if not self.output_height > 0:
            raise ValueError('inpt height smaller than pool height')
        if not self.output_width > 0:
            raise ValueError('inpt width smaller than pool width')

        self.n_output = n_output

        super(MaxPool2d, self).__init__(declare=declare, name=name)

    def _forward(self):
        self.output_in = downsample.max_pool_2d(
            input=self.inpt,
            ds=(self.pool_height, self.pool_width),
            st=self.st,
            ignore_border=True
        )

        f = lookup(self.transfer, _transfer)
        self.output = f(self.output_in)


class Upsample2d(Layer):

    def __init__(self, inpt, inpt_height, inpt_width,
                 upsample_height, upsample_width,
                 n_output,
                 padding=None,
                 transfer='identity',
                 declare=None, name=None):

        self.inpt = inpt
        self.inpt_height = inpt_height
        self.inpt_width = inpt_width
        self.upsample_height = upsample_height
        self.upsample_width = upsample_width
        self.transfer = transfer

        if upsample_height != upsample_height:
            raise ValueError("upsample height and upsample width are different, not supported yet")

        self.upsample = upsample_height

        if padding is None:
            self.padding_left = self.padding_right = self.padding_top = self.padding_bottom = 0
        elif type(padding) == int:
            self.padding_left = self.padding_right = self.padding_top = self.padding_bottom = padding
        elif len(padding) == 2:
            self.padding_left = self.padding_right = padding[0]
            self.padding_top = self.padding_bottom = padding[1]
        elif len(padding) == 4:
            self.padding_left = padding[0]
            self.padding_top = padding[1]
            self.padding_right = padding[2]
            self.padding_bottom = padding[3]
        else:
            raise ValueError("padding is not set properly (either None, int, (int, int), (int, int, int, int))")

        #self.output_height = inpt_height * upsample_height + self.padding_left + self.padding_right
        #self.output_width = inpt_width * upsample_width + self.padding_top + self.padding_bottom
        self.output_height = upsample_height
        self.output_width = upsample_width

        print(self.output_height, self.output_width, self.padding_top, self.inpt_height * self.upsample_height, self.padding_left, self.inpt_width * self.upsample_width)

        self.n_output = n_output

        super(Upsample2d, self).__init__(declare=declare, name=name)

    def _forward(self):

        # repeat = T.extra_ops.repeat(
        #     T.extra_ops.repeat(self.inpt, self.upsample_height, axis=2),
        #     self.upsample_width, axis=3
        # )
        #upsamp = self.inpt.repeat(self.upsample_height, axis=2).repeat(self.upsample_width, axis=3)

        inpt_shape = self.inpt.shape
        #output_shape = (inpt_shape[0], inpt_shape[1], inpt_shape[2] * self.upsample_height, inpt_shape[3] * self.upsample_width)
        output_shape = (inpt_shape[0], inpt_shape[1], self.upsample_height, self.upsample_width)

        #in_dim = inpt_shape[2] * inpt_shape[3]
        #out_dim = output_shape[2] * output_shape[3]

        #print(in_dim * out_dim)

        #upsamp_matrix = T.alloc(0., in_dim, out_dim)
        #rows = T.arange(in_dim)
        #cols = rows * self.upsample + (rows / inpt_shape[2] * self.upsample * inpt_shape[3])
        #upsamp_matrix = T.set_subtensor(upsamp_matrix[rows, cols], 1.)
        new_matrix = T.alloc(0.0, output_shape[0], output_shape[1], output_shape[2], output_shape[3])
        new_matrix = T.set_subtensor(new_matrix[:, :, ::2, ::2], self.inpt)


        #flat = self.inpt.reshape((inpt_shape[0], inpt_shape[1], inpt_shape[2] * inpt_shape[3]))

        #upsamp_flat = T.dot(flat, upsamp_matrix)

        #upsamp = upsamp_flat.reshape(output_shape)

        if self.padding_left == 0 and self.padding_top == 0:
            self.output_in = new_matrix
        else:
            self.output_in = T.alloc(0., self.inpt.shape[0], self.inpt.shape[1], self.output_height, self.output_width)

            self.output_in = T.set_subtensor(
                self.output_in[
                    :,
                    :,
                    self.padding_top:self.padding_top + self.inpt_height * self.upsample_height,
                    self.padding_left:self.padding_left + self.inpt_width * self.upsample_width
                ],
                new_matrix
            )

        f = lookup(self.transfer, _transfer)
        self.output = f(self.output_in)
        #self.output = self.output_in
        #self.output = f(new_matrix)

class ParametricReLu(Layer):
    def __init__(self, inpt, inpt_height, inpt_width, n_channel,
                 shared=False,
                 declare=None, name=None):
        self.inpt = inpt
        self.inpt_height = inpt_height
        self.inpt_width = inpt_width
        self.n_channel = n_channel

        self.output_height = inpt_height
        self.output_width = inpt_width
        self.n_output = n_channel

        self.shared = shared

        super(ParametricReLu, self).__init__(declare=declare, name=name)

    def _forward(self):

        shape = (1, 1, 1, 1) if self.shared else (1, self.n_channel, 1, 1)
        shared_axes = (0, 1, 2, 3) if self.shared else (0, 2, 3)
        self.alpha = self.declare(shape)
        alpha = T.addbroadcast(self.alpha, *shared_axes)
        self.output = T.nnet.relu(self.inpt, alpha)
