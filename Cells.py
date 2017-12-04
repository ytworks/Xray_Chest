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



def inception_cell(x,
                   Act = 'Relu',
                   InputNode = [64, 64, 256],
                   Channels = 32,
                   Strides0 = [1, 1, 1, 1],
                   Initializer = 'He',
                   vname = 'Inception',
                   regularization = False,
                   SE = True,
                   Big = False):
    # 1st Layers
    BN = False
    Act_Internal = 'Equal'
    x = Layers.batch_normalization(x = x, shape = [InputNode[2]])
    x = AF.select_activation(Act)(x)
    x01 = Layers.convolution2d(x = x,
                               FilterSize = [1, 1, InputNode[2], Channels],
                               Initializer = Initializer,
                               Strides = [Strides0[1], Strides0[2]],
                               Padding = 'SAME',
                               ActivationFunction = Act_Internal,
                               BatchNormalization = BN,
                               Regularization = regularization,
                               vname = vname + '_Conv_01')

    x02 = Layers.convolution2d(x = x,
                               FilterSize = [3, 3, InputNode[2], Channels],
                               Initializer = Initializer,
                               Strides = [Strides0[1], Strides0[2]],
                               Padding = 'SAME',
                               ActivationFunction = Act_Internal,
                               BatchNormalization = BN,
                               Regularization = regularization,
                               vname = vname + '_Conv_02')

    x03 = Layers.convolution2d(x = x,
                               FilterSize = [3, 1, InputNode[2], Channels],
                               Initializer = Initializer,
                               Strides = [Strides0[1], Strides0[2]],
                               Padding = 'SAME',
                               ActivationFunction = Act_Internal,
                               BatchNormalization = BN,
                               Regularization = regularization,
                               vname = vname + '_Conv_03')

    x04 = Layers.convolution2d(x = x,
                               FilterSize = [1, 3, InputNode[2], Channels],
                               Initializer = Initializer,
                               Strides = [Strides0[1], Strides0[2]],
                               Padding = 'SAME',
                               ActivationFunction = Act_Internal,
                               BatchNormalization = BN,
                               Regularization = regularization,
                               vname = vname + '_Conv_04')

    x05 = Layers.convolution2d(x = x,
                               FilterSize = [5, 5, InputNode[2], Channels],
                               Initializer = Initializer,
                               Strides = [Strides0[1], Strides0[2]],
                               Padding = 'SAME',
                               ActivationFunction = Act_Internal,
                               BatchNormalization = BN,
                               Regularization = regularization,
                               vname = vname + '_Conv_05')

    x06 = Layers.convolution2d(x = x,
                               FilterSize = [5, 1, InputNode[2], Channels],
                               Initializer = Initializer,
                               Strides = [Strides0[1], Strides0[2]],
                               Padding = 'SAME',
                               ActivationFunction = Act_Internal,
                               BatchNormalization = BN,
                               Regularization = regularization,
                               vname = vname + '_Conv_06')

    x07 = Layers.convolution2d(x = x,
                               FilterSize = [1, 5, InputNode[2], Channels],
                               Initializer = Initializer,
                               Strides = [Strides0[1], Strides0[2]],
                               Padding = 'SAME',
                               ActivationFunction = Act_Internal,
                               BatchNormalization = BN,
                               Regularization = regularization,
                               vname = vname + '_Conv_07')
    p01 = Layers.pooling(x = x,
                              ksize=[2, 2],
                              strides=[1, 1],
                              padding='SAME',
                              algorithm = 'Avg')
    pc1 = Layers.convolution2d(x = p01,
                               FilterSize = [1, 1, InputNode[2], Channels],
                               Initializer = Initializer,
                               Strides = [Strides0[1], Strides0[2]],
                               Padding = 'SAME',
                               ActivationFunction = Act_Internal,
                               BatchNormalization = BN,
                               Regularization = regularization,
                               vname = vname + '_Conv_P01')
    p02 = Layers.pooling(x = x,
                              ksize=[2, 2],
                              strides=[1, 1],
                              padding='SAME',
                              algorithm = 'Max')
    pc2 = Layers.convolution2d(x = p02,
                               FilterSize = [1, 1, InputNode[2], Channels],
                               Initializer = Initializer,
                               Strides = [Strides0[1], Strides0[2]],
                               Padding = 'SAME',
                               ActivationFunction = Act_Internal,
                               BatchNormalization = BN,
                               Regularization = regularization,
                               vname = vname + '_Conv_P02')

    if Big:
        x0b = Layers.convolution2d(x = x,
                                   FilterSize = [7, 7, InputNode[2], Channels],
                                   Initializer = Initializer,
                                   Strides = [Strides0[1], Strides0[2]],
                                   Padding = 'SAME',
                                   ActivationFunction = Act_Internal,
                                   BatchNormalization = BN,
                                   Regularization = regularization,
                                   vname = vname + '_Conv_08')
        x0bv = Layers.convolution2d(x = x,
                                   FilterSize = [7, 1, InputNode[2], Channels],
                                   Initializer = Initializer,
                                   Strides = [Strides0[1], Strides0[2]],
                                   Padding = 'SAME',
                                   ActivationFunction = Act_Internal,
                                   BatchNormalization = BN,
                                   Regularization = regularization,
                                   vname = vname + '_Conv_09')
        x0bh = Layers.convolution2d(x = x,
                                   FilterSize = [1, 7, InputNode[2], Channels],
                                   Initializer = Initializer,
                                   Strides = [Strides0[1], Strides0[2]],
                                   Padding = 'SAME',
                                   ActivationFunction = Act_Internal,
                                   BatchNormalization = BN,
                                   Regularization = regularization,
                                   vname = vname + '_Conv_10')

    y01 = Layers.concat(xs = [x01, x02, x03, x04, x05, x06, x07, pc1, pc2],
                        concat_type = 'Channel')

    if Big:
        y02 = Layers.concat(xs = [y01, x0b, x0bh, x0bh],
                            concat_type = 'Channel')
        if SE:
            y01 = SE_module(x = y01,
                            Act = 'Relu',
                            InputNode =[InputNode[0], InputNode[1], 13 * Channels],
                            vname = vname + '_SE')
        return y02
    else:
        if SE:
            y01 = SE_module(x = y01,
                            Act = 'Relu',
                            InputNode =[InputNode[0], InputNode[1], 9 * Channels],
                            vname = vname + '_SE')
        return y01


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
                    Initializer = 'Xavier',
                    ActivationFunction = 'Sigmoid',
                    MaxoutSize = 3,
                    BatchNormalization = False,
                    Regularization = False,
                    vname = vname + '_FNN1')

    x4 = Layers.reshape_tensor(x = x3,
                               shape = [1, 1, InputNode[2]])
    scale = x * x4
    return scale


# Inception Res module
def inception_res_cell(x,
                        Act = 'Relu',
                        InputNode = [64, 64, 16 * 7],
                        Channels = [16, 16 * 7],
                        Strides0 = [1, 1, 1, 1],
                        Strides1 = [1, 1, 1, 1],
                        Initializer = 'He',
                        Regularization = False,
                        vname = 'Res',
                        Big = False,
                        SE = True):
    if Big:
        parallels = 12
    else:
        parallels = 9
    x01 = inception_cell(x = x,
                         Act = Act,
                         InputNode = InputNode,
                         Channels = Channels[0],
                         Strides0 = Strides0,
                         Initializer = Initializer,
                         vname = vname + '_inception1',
                         regularization = Regularization,
                         Big = Big,
                         SE = SE)
    x02 = inception_cell(x = x01,
                         Act = 'Equal',
                         InputNode = [InputNode[0] / Strides0[1], InputNode[1] / Strides0[2], Channels[1] * parallels],
                         Channels = Channels[1],
                         Strides0 = Strides1,
                         Initializer = Initializer,
                         vname = vname + '_inception2',
                         regularization = Regularization,
                         Big = Big,
                         SE = SE
                         )
    # ダウンサンプリングの場合
    if Strides0 == [1, 1, 1, 1]:
        shortcut = x
    else:
        print("Inception-Res cell: Down Sampling")
        shortcut = Layers.pooling(x = x,
                                  ksize=[Strides0[1], Strides0[2]],
                                  strides=[Strides0[1], Strides0[2]],
                                  padding='VALID',
                                  algorithm = 'Avg')
    # チャネルが一致しない場合
    if InputNode[2] != Channels[1] * parallels:
        print("Inception-Res cell: Synchronizing Channel Number")
        sc  = Layers.convolution2d(x = shortcut,
                                   FilterSize = [1, 1, InputNode[2], Channels[1] * parallels],
                                   Initializer = Initializer,
                                   Strides = [Strides1[1], Strides1[2]],
                                   Padding = 'SAME',
                                   ActivationFunction = 'Equal',
                                   BatchNormalization = True,
                                   Regularization = Regularization,
                                   vname = vname + '_Conv_02')
    else:
        sc = shortcut
    with tf.variable_scope(vname + '_Act') as scope:
        x03 = AF.select_activation(Act)(x02 + sc)
    if SE:
        x03 = SE_module(x = x03,
                        InputNode = [InputNode[0] / Strides0[1], InputNode[1] / Strides0[2], Channels[1] * parallels],
                        Act = 'Relu',
                        Rate = 0.5,
                        vname = vname + '_SE_01')
    return x03
