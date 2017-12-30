#! /usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
from LinearMotor import Variables
from LinearMotor import ActivationFunctions as AF
from LinearMotor import Layers
from LinearMotor import Outputs


# Stem cell
def stem_cell(x,
              InputNode = [512, 512, 3],
              Channels = 128,
              Initializer = 'He',
              vname = 'Stem',
              regularization = False,
              Training = True):
    x01 = Layers.convolution2d(x = x,
                               FilterSize = [7, 7, InputNode[2], Channel],
                               Initializer = Initializer,
                               Strides = [2, 2],
                               Padding = 'SAME',
                               ActivationFunction = 'Equal',
                               BatchNormalization = False,
                               Renormalization = False,
                               Training = Training,
                               Regularization = regularization,
                               vname = vname + '_Conv_01')
    p_max = Layers.pooling(x = x01,
                           ksize=[3, 3],
                           strides=[2, 2],
                           padding='SAME',
                           algorithm = 'Max')
    return p_max


# Conv Block
def conv_block(x,
               Act = 'Relu',
               InputNode = [64, 64, 32],
               Strides = [1, 1, 1, 1],
               Renormalization = False,
               Regularization = False,
               rmax = None,
               dmax = None,
               SE = True,
               Training = True,
               vname = '_ConvBlock'):
    # Batch Normalization
    x_bn1 = Layers.batch_normalization(x = x,
                                       shape = InputNode[2],
                                       vname = vname + '_BN01',
                                       dim = [0, 1, 2],
                                       Renormalization = Renormalization,
                                       Training = Training,
                                       rmax = rmax,
                                       dmax = dmax)
    # Activation Function
    with tf.variable_scope(vname + '_Act01') as scope:
        x_act1 = AF.select_activation(Act)(x_bn1)

    x01 = Layers.convolution2d(x = x_act1,
                               FilterSize = [1, 1, InputNode[2], InputNode[2] * 4],
                               Initializer = Initializer,
                               Strides = [Strides[1], Strides[2]],
                               Padding = 'SAME',
                               ActivationFunction = 'Equal',
                               BatchNormalization = False,
                               Renormalization = False,
                               Training = Training,
                               Regularization = Regularization,
                               vname = vname + '_Conv_01')
    if SE:
        x01 = SE_module(x = x01,
                        InputNode = [InputNode[0], InputNode[1], InputNode[2] * 4],
                        Act = 'Relu',
                        vname = vname + '_SE01')

    # Batch Normalization
    x_bn2 = Layers.batch_normalization(x = x01,
                                      shape = InputNode[2] * 4,
                                      vname = vname + '_BN02',
                                      dim = [0, 1, 2],
                                      Renormalization = Renormalization,
                                      Training = Training,
                                      rmax = rmax,
                                      dmax = dmax)
    # Activation Function
    with tf.variable_scope(vname + '_Act02') as scope:
        x_act2 = AF.select_activation(Act)(x_bn2)

    x02 = Layers.convolution2d(x = x_act2,
                               FilterSize = [3, 3, InputNode[2] * 4, InputNode[2]],
                               Initializer = Initializer,
                               Strides = [Strides[1], Strides[2]],
                               Padding = 'SAME',
                               ActivationFunction = 'Equal',
                               BatchNormalization = False,
                               Renormalization = False,
                               Training = Training,
                               Regularization = Regularization,
                               vname = vname + '_Conv_02')
    if SE:
        x02 = SE_module(x = x02,
                        InputNode = [InputNode[0], InputNode[1], InputNode[2]],
                        Act = 'Relu',
                        vname = vname + '_SE02')
    return x02





# resnet cell
def inception_res_cell(x,
                       Act = 'Relu',
                       InputNode = [64, 64, 16 * 7],
                       Channels0 = [2, 2, 2, 2, 2, 2],
                       Channels1 = [2, 2, 2, 2, 2, 2],
                       Strides0 = [1, 1, 1, 1],
                       Strides1 = [1, 1, 1, 1],
                       Initializer = 'He',
                       Regularization = False,
                       Renormalization = True,
                       Rmax = None,
                       Dmax = None,
                       vname = 'Res',
                       SE = True,
                       Training = True):
    # inception1
    x01 = inception_cell(x = x,
                         Act = Act,
                         InputNode = InputNode,
                         Channels = Channels0,
                         Strides0 = Strides0,
                         Initializer = Initializer,
                         vname = vname + 'Inception01',
                         regularization = Regularization,
                         renormalization = Renormalization,
                         rmax = Rmax,
                         dmax = Dmax,
                         SE = SE,
                         Training = Training)

    # inception2
    x02 = inception_cell(x = x01,
                         Act = Act,
                         InputNode = [InputNode[0], InputNode[1], sum(Channels0)],
                         Channels = Channels1,
                         Strides0 = Strides0,
                         Initializer = Initializer,
                         vname = vname + 'Inception02',
                         regularization = Regularization,
                         renormalization = Renormalization,
                         rmax = Rmax,
                         dmax = Dmax,
                         SE = SE,
                         Training = Training)
    # チャネルが一致しない場合
    if InputNode[2] != sum(Channels1):
        print("Inception-Res cell: Synchronizing Channel Number")
        sc  = Layers.convolution2d(x = x,
                                   FilterSize = [1, 1, InputNode[2], sum(Channels1)],
                                   Initializer = Initializer,
                                   Strides = [Strides1[1], Strides1[2]],
                                   Padding = 'SAME',
                                   ActivationFunction = 'Equal',
                                   BatchNormalization = False,
                                   Renormalization = False,
                                   Regularization = Regularization,
                                   Training = Training,
                                   vname = vname + '_Conv_SC')
    else:
        sc = x

    # merge
    return x02 + sc


# Inception cell
def inception_cell(x,
                   Act = 'Relu',
                   InputNode = [64, 64, 256],
                   Channels = [2, 2, 2, 2, 2, 2],
                   Strides0 = [1, 1, 1, 1],
                   Initializer = 'He',
                   vname = 'Inception',
                   regularization = False,
                   renormalization = True,
                   rmax = None,
                   dmax = None,
                   SE = True,
                   Training = True):
    BN = False
    Renorm = False
    # Batch Normalization
    x_bn = Layers.batch_normalization(x = x,
                                      shape = InputNode[2],
                                      vname = vname + '_BN',
                                      dim = [0, 1, 2],
                                      Renormalization = renormalization,
                                      Training = Training,
                                      rmax = rmax,
                                      dmax = dmax)
    # Activation Function
    with tf.variable_scope(vname + '_Act') as scope:
        x_act = AF.select_activation(Act)(x_bn)

    # Inception
    ## 3x3
    x01 = Layers.convolution2d(x = x_act,
                               FilterSize = [3, 3, InputNode[2], Channels[0]],
                               Initializer = Initializer,
                               Strides = [Strides0[1], Strides0[2]],
                               Padding = 'SAME',
                               ActivationFunction = 'Equal',
                               BatchNormalization = BN,
                               Renormalization = Renorm,
                               Training = Training,
                               Regularization = regularization,
                               vname = vname + '_Conv_01')
    ## 3x1
    x02 = Layers.convolution2d(x = x_act,
                               FilterSize = [3, 1, InputNode[2], Channels[1]],
                               Initializer = Initializer,
                               Strides = [Strides0[1], Strides0[2]],
                               Padding = 'SAME',
                               ActivationFunction = 'Equal',
                               BatchNormalization = BN,
                               Renormalization = Renorm,
                               Training = Training,
                               Regularization = regularization,
                               vname = vname + '_Conv_02')
    ## 1x3
    x03 = Layers.convolution2d(x = x_act,
                               FilterSize = [1, 3, InputNode[2], Channels[2]],
                               Initializer = Initializer,
                               Strides = [Strides0[1], Strides0[2]],
                               Padding = 'SAME',
                               ActivationFunction = 'Equal',
                               BatchNormalization = BN,
                               Renormalization = Renorm,
                               Training = Training,
                               Regularization = regularization,
                               vname = vname + '_Conv_03')
    ## Avg 1x1
    p_avg = Layers.pooling(x = x_act,
                           ksize=[2, 2],
                           strides=[1, 1],
                           padding='SAME',
                           algorithm = 'Avg')
    x04 = Layers.convolution2d(x = p_avg,
                               FilterSize = [1, 1, InputNode[2], Channels[3]],
                               Initializer = Initializer,
                               Strides = [Strides0[1], Strides0[2]],
                               Padding = 'SAME',
                               ActivationFunction = 'Equal',
                               BatchNormalization = BN,
                               Renormalization = Renorm,
                               Training = Training,
                               Regularization = regularization,
                               vname = vname + '_Conv_04')
    ## Max 1x1
    p_max = Layers.pooling(x = x_act,
                           ksize=[2, 2],
                           strides=[1, 1],
                           padding='SAME',
                           algorithm = 'Max')
    x05 = Layers.convolution2d(x = p_max,
                               FilterSize = [1, 1, InputNode[2], Channels[4]],
                               Initializer = Initializer,
                               Strides = [Strides0[1], Strides0[2]],
                               Padding = 'SAME',
                               ActivationFunction = 'Equal',
                               BatchNormalization = BN,
                               Renormalization = Renorm,
                               Training = Training,
                               Regularization = regularization,
                               vname = vname + '_Conv_05')
    ## 1x1
    x06 = Layers.convolution2d(x = x_act,
                               FilterSize = [1, 1, InputNode[2], Channels[5]],
                               Initializer = Initializer,
                               Strides = [Strides0[1], Strides0[2]],
                               Padding = 'SAME',
                               ActivationFunction = 'Equal',
                               BatchNormalization = BN,
                               Renormalization = Renorm,
                               Training = Training,
                               Regularization = regularization,
                               vname = vname + '_Conv_06')

    # merge
    x_concat = Layers.concat(xs = [x01, x02, x03, x04, x05, x06],
                             concat_type = 'Channel')
    # SE
    if SE:
        x_concat = SE_module(x = x_concat,
                             InputNode = [InputNode[0], InputNode[1], sum(Channels)],
                             Act = 'Relu',
                             vname = vname + '_SE')
    return x_concat


# SE cell
def SE_module(x,
              InputNode,
              Act = 'Relu',
              Rate = 0.5,
              vname = 'SE'):
    # Global Average Pooling
    x0 = Layers.pooling(x = x,
                        ksize=[InputNode[0], InputNode[1]],
                        strides=[InputNode[0], InputNode[1]],
                        padding='SAME',
                        algorithm = 'Avg')

    x1 = Layers.reshape_tensor(x = x0,
                               shape = [1 * 1 * InputNode[2]])

    x2 = Layers.fnn(x = x1,
                    InputSize = InputNode[2],
                    OutputSize = int(InputNode[2] * Rate),
                    Initializer = 'He' if Act == 'Relu' else 'Xavier',
                    ActivationFunction = Act,
                    MaxoutSize = 3,
                    BatchNormalization = False,
                    Regularization = False,
                    vname = vname + '_FNN0')

    x3 = Layers.fnn(x = x2,
                    InputSize = int(InputNode[2] * Rate),
                    OutputSize = InputNode[2],
                    Initializer = 'Xavier_normal',
                    ActivationFunction = 'Sigmoid',
                    MaxoutSize = 3,
                    BatchNormalization = False,
                    Regularization = False,
                    vname = vname + '_FNN1')

    x4 = Layers.reshape_tensor(x = x3,
                               shape = [1, 1, InputNode[2]])
    scale = x * x4
    return scale
