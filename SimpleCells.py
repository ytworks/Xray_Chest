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
                               FilterSize = [7, 7, InputNode[2], Channels],
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


def densenet(x,
             root,
             Act = 'Relu',
             GrowthRate = 12,
             InputNode = [64, 64, 32],
             Initializer = 'He',
             Strides = [1, 1, 1, 1],
             Renormalization = False,
             Regularization = False,
             rmax = None,
             dmax = None,
             SE = True,
             Training = True,
             vname = '_DenseNet'):

    root0 = tf.image.resize_images(images = root,
                                   size = [InputNode[0], InputNode[1]],
                                   method=tf.image.ResizeMethod.BICUBIC,
                                   align_corners=False)
    x_plus_root = Layers.concat(xs = [x, root0], concat_type = 'Channel')

    x01 = dense_cell(x = x_plus_root,
                     Act = Act,
                     GrowthRate = GrowthRate,
                     InputNode = [InputNode[0], InputNode[1], InputNode[2] + 3],
                     Initializer = Initializer,
                     Strides = Strides,
                     Renormalization = Renormalization,
                     Regularization = Regularization,
                     rmax = rmax,
                     dmax = dmax,
                     SE = SE,
                     Training = Training,
                     vname = vname + '_Dense01')
    x02 = transition_cell(x = x01,
                          Act = Act,
                          InputNode = [InputNode[0], InputNode[1], InputNode[2] + 3 + GrowthRate * 4],
                          Initializer = Initializer,
                          Strides = Strides,
                          Renormalization = Renormalization,
                          Regularization = Regularization,
                          rmax = rmax,
                          dmax = dmax,
                          SE = SE,
                          Training = Training,
                          vname = vname +'_Transition01')

    root1 = tf.image.resize_images(images = root,
                                   size = [InputNode[0]/2, InputNode[1]/2],
                                   method=tf.image.ResizeMethod.BICUBIC,
                                   align_corners=False)
    x02_plus_root = Layers.concat(xs = [x02, root1], concat_type = 'Channel')

    x03 = dense_cell(x = x02_plus_root,
                     Act = Act,
                     GrowthRate = GrowthRate,
                     InputNode = [InputNode[0] / 2, InputNode[1] / 2, InputNode[2] + 6 + GrowthRate * 4],
                     Initializer = Initializer,
                     Strides = Strides,
                     Renormalization = Renormalization,
                     Regularization = Regularization,
                     rmax = rmax,
                     dmax = dmax,
                     SE = SE,
                     Training = Training,
                     vname = vname + '_Dense02')
    x04 = transition_cell(x = x03,
                          Act = Act,
                          InputNode = [InputNode[0] / 2, InputNode[1] / 2, InputNode[2] + 6 + GrowthRate * 8],
                          Initializer = Initializer,
                          Strides = Strides,
                          Renormalization = Renormalization,
                          Regularization = Regularization,
                          rmax = rmax,
                          dmax = dmax,
                          SE = SE,
                          Training = Training,
                          vname = vname +'_Transition02')

    root2 = tf.image.resize_images(images = root,
                                   size = [InputNode[0]/4, InputNode[1]/4],
                                   method=tf.image.ResizeMethod.BICUBIC,
                                   align_corners=False)
    x04_plus_root = Layers.concat(xs = [x04, root2], concat_type = 'Channel')


    x05 = dense_cell(x = x04_plus_root,
                     Act = Act,
                     GrowthRate = GrowthRate,
                     InputNode = [InputNode[0] / 4, InputNode[1] / 4, InputNode[2] + 9 + GrowthRate * 8],
                     Initializer = Initializer,
                     Strides = Strides,
                     Renormalization = Renormalization,
                     Regularization = Regularization,
                     rmax = rmax,
                     dmax = dmax,
                     SE = SE,
                     Training = Training,
                     vname = vname + '_Dense03')
    x06 = transition_cell(x = x05,
                          Act = Act,
                          InputNode = [InputNode[0] / 4, InputNode[1] / 4, InputNode[2] + 9 + GrowthRate * 12],
                          Initializer = Initializer,
                          Strides = Strides,
                          Renormalization = Renormalization,
                          Regularization = Regularization,
                          rmax = rmax,
                          dmax = dmax,
                          SE = SE,
                          Training = Training,
                          vname = vname +'_Transition03')

    root3 = tf.image.resize_images(images = root,
                                   size = [InputNode[0]/8, InputNode[1]/8],
                                   method=tf.image.ResizeMethod.BICUBIC,
                                   align_corners=False)
    x06_plus_root = Layers.concat(xs = [x06, root3], concat_type = 'Channel')

    x07 = dense_cell(x = x06_plus_root,
                     Act = Act,
                     GrowthRate = GrowthRate,
                     InputNode = [InputNode[0] / 8, InputNode[1] / 8, InputNode[2] + 12 + GrowthRate * 12],
                     Initializer = Initializer,
                     Strides = Strides,
                     Renormalization = Renormalization,
                     Regularization = Regularization,
                     rmax = rmax,
                     dmax = dmax,
                     SE = SE,
                     Training = Training,
                     vname = vname + '_Dense04')
    x08 = transition_cell(x = x07,
                          Act = Act,
                          InputNode = [InputNode[0] / 8, InputNode[1] / 8, InputNode[2] + 12 + GrowthRate * 16],
                          Initializer = Initializer,
                          Strides = Strides,
                          Renormalization = Renormalization,
                          Regularization = Regularization,
                          rmax = rmax,
                          dmax = dmax,
                          SE = SE,
                          Training = Training,
                          vname = vname +'_Transition04')
    # Batch Normalization
    '''
    x_bn1 = Layers.batch_normalization(x = x08,
                                       shape = InputNode[2] + 12 + GrowthRate * 16,
                                       vname = vname + '_BN01',
                                       dim = [0, 1, 2],
                                       Renormalization = Renormalization,
                                       Training = Training,
                                       rmax = rmax,
                                       dmax = dmax)
    '''
    x_bn1 = Layers.group_normalization(x = x08, G = 16,
                                       eps = 1e-5, vname = vanme + '_GN01')
    # Activation Function
    with tf.variable_scope(vname + '_Act01') as scope:
        x_act1 = AF.select_activation(Act)(x_bn1)
    return x_act1



def dense_cell(x,
               Act = 'Relu',
               GrowthRate = 12,
               InputNode = [64, 64, 32],
               Initializer = 'He',
               Strides = [1, 1, 1, 1],
               Renormalization = False,
               Regularization = False,
               rmax = None,
               dmax = None,
               SE = True,
               Training = True,
               vname = '_Dense'):

    x01 = conv_block(x = x,
                     Act = Act,
                     GrowthRate = GrowthRate,
                     InputNode = [InputNode[0], InputNode[1], InputNode[2]],
                     Initializer = Initializer,
                     Strides = [1, 1, 1, 1],
                     Renormalization = Renormalization,
                     Regularization = Regularization,
                     rmax = rmax,
                     dmax = dmax,
                     SE = SE,
                     Training = Training,
                     vname = vname + '_ConvBlock01')
    x02 = Layers.concat(xs = [x, x01], concat_type = 'Channel')
    if SE:
        x02 = SE_module(x = x02,
                        InputNode = [InputNode[0], InputNode[1], InputNode[2] + GrowthRate],
                        Act = Act,
                        vname = vname + '_SE01')

    x11 = conv_block(x = x02,
                     Act = Act,
                     GrowthRate = GrowthRate,
                     InputNode = [InputNode[0], InputNode[1], InputNode[2] + GrowthRate],
                     Initializer = Initializer,
                     Strides = [1, 1, 1, 1],
                     Renormalization = Renormalization,
                     Regularization = Regularization,
                     rmax = rmax,
                     dmax = dmax,
                     SE = SE,
                     Training = Training,
                     vname = vname +'_ConvBlock02')
    x12 = Layers.concat(xs = [x, x01, x11], concat_type = 'Channel')
    if SE:
        x12 = SE_module(x = x12,
                        InputNode = [InputNode[0], InputNode[1], InputNode[2] + GrowthRate * 2],
                        Act = Act,
                        vname = vname + '_SE02')

    x21 = conv_block(x = x12,
                     Act = Act,
                     GrowthRate = GrowthRate,
                     InputNode = [InputNode[0], InputNode[1], InputNode[2] + GrowthRate * 2],
                     Initializer = Initializer,
                     Strides = [1, 1, 1, 1],
                     Renormalization = Renormalization,
                     Regularization = Regularization,
                     rmax = rmax,
                     dmax = dmax,
                     SE = SE,
                     Training = Training,
                     vname = vname +'_ConvBlock03')
    x22 = Layers.concat(xs = [x, x01, x11, x21], concat_type = 'Channel')
    if SE:
        x22 = SE_module(x = x22,
                        InputNode = [InputNode[0], InputNode[1], InputNode[2] + GrowthRate * 3],
                        Act = Act,
                        vname = vname + '_SE03')

    x31 = conv_block(x = x22,
                     Act = Act,
                     GrowthRate = GrowthRate,
                     InputNode = [InputNode[0], InputNode[1], InputNode[2] + GrowthRate * 3],
                     Initializer = Initializer,
                     Strides = [1, 1, 1, 1],
                     Renormalization = Renormalization,
                     Regularization = Regularization,
                     rmax = rmax,
                     dmax = dmax,
                     SE = SE,
                     Training = Training,
                     vname = vname +'_ConvBlock04')
    x32 = Layers.concat(xs = [x, x01, x11, x21, x31], concat_type = 'Channel')
    if SE:
        x32 = SE_module(x = x32,
                        InputNode = [InputNode[0], InputNode[1], InputNode[2] + GrowthRate * 4],
                        Act = Act,
                        vname = vname + '_SE04')
    return x32






# Conv Block
def conv_block(x,
               Act = 'Relu',
               InputNode = [64, 64, 32],
               Initializer = 'He',
               GrowthRate = 12,
               Strides = [1, 1, 1, 1],
               Renormalization = False,
               Regularization = False,
               rmax = None,
               dmax = None,
               SE = True,
               Training = True,
               vname = '_ConvBlock'):
    # Batch Normalization
    '''
    x_bn1 = Layers.batch_normalization(x = x,
                                       shape = InputNode[2],
                                       vname = vname + '_BN01',
                                       dim = [0, 1, 2],
                                       Renormalization = Renormalization,
                                       Training = Training,
                                       rmax = rmax,
                                       dmax = dmax)
    '''
    x_bn1 = Layers.group_normalization(x = x, G = 16,
                                       eps = 1e-5, vname = vanme + '_GN01')
    # Activation Function
    with tf.variable_scope(vname + '_Act01') as scope:
        x_act1 = AF.select_activation(Act)(x_bn1)

    x0a = Layers.convolution2d(x = x_act1,
                               FilterSize = [1, 1, InputNode[2], GrowthRate * 2],
                               Initializer = Initializer,
                               Strides = [Strides[1], Strides[2]],
                               Padding = 'SAME',
                               ActivationFunction = 'Equal',
                               BatchNormalization = False,
                               Renormalization = False,
                               Training = Training,
                               Regularization = Regularization,
                               vname = vname + '_Conv_01a')
    p_avg = Layers.pooling(x = x_act1,
                           ksize=[2, 2],
                           strides=[1, 1],
                           padding='SAME',
                           algorithm = 'Avg')
    x0b = Layers.convolution2d(x = p_avg,
                               FilterSize = [1, 1, InputNode[2], GrowthRate * 1],
                               Initializer = Initializer,
                               Strides = [Strides[1], Strides[2]],
                               Padding = 'SAME',
                               ActivationFunction = 'Equal',
                               BatchNormalization = False,
                               Renormalization = False,
                               Training = Training,
                               Regularization = Regularization,
                               vname = vname + '_Conv_01b')
    p_max = Layers.pooling(x = x_act1,
                           ksize=[2, 2],
                           strides=[1, 1],
                           padding='SAME',
                           algorithm = 'Max')
    x0c = Layers.convolution2d(x = p_max,
                               FilterSize = [1, 1, InputNode[2], GrowthRate * 1],
                               Initializer = Initializer,
                               Strides = [Strides[1], Strides[2]],
                               Padding = 'SAME',
                               ActivationFunction = 'Equal',
                               BatchNormalization = False,
                               Renormalization = False,
                               Training = Training,
                               Regularization = Regularization,
                               vname = vname + '_Conv_01c')
    x01 = Layers.concat(xs = [x0a, x0b, x0c], concat_type = 'Channel')
    if SE:
        x01 = SE_module(x = x01,
                        InputNode = [InputNode[0], InputNode[1], GrowthRate * 4],
                        Act = Act,
                        vname = vname + '_SE01')

    # Batch Normalization
    '''
    x_bn2 = Layers.batch_normalization(x = x01,
                                      shape = GrowthRate * 4,
                                      vname = vname + '_BN02',
                                      dim = [0, 1, 2],
                                      Renormalization = Renormalization,
                                      Training = Training,
                                      rmax = rmax,
                                      dmax = dmax)
    '''
    x_bn2 = Layers.group_normalization(x = x01, G = 16,
                                       eps = 1e-5, vname = vanme + '_GN02')
    # Activation Function
    with tf.variable_scope(vname + '_Act02') as scope:
        x_act2 = AF.select_activation(Act)(x_bn2)

    x1a = Layers.convolution2d(x = x_act2,
                               FilterSize = [3, 3, GrowthRate * 4, GrowthRate/2],
                               Initializer = Initializer,
                               Strides = [Strides[1], Strides[2]],
                               Padding = 'SAME',
                               ActivationFunction = 'Equal',
                               BatchNormalization = False,
                               Renormalization = False,
                               Training = Training,
                               Regularization = Regularization,
                               vname = vname + '_Conv_02a')
    x1b = Layers.convolution2d(x = x_act2,
                               FilterSize = [3, 1, GrowthRate * 4, GrowthRate/4],
                               Initializer = Initializer,
                               Strides = [Strides[1], Strides[2]],
                               Padding = 'SAME',
                               ActivationFunction = 'Equal',
                               BatchNormalization = False,
                               Renormalization = False,
                               Training = Training,
                               Regularization = Regularization,
                               vname = vname + '_Conv_02b')
    x1c = Layers.convolution2d(x = x_act2,
                               FilterSize = [1, 3, GrowthRate * 4, GrowthRate/4],
                               Initializer = Initializer,
                               Strides = [Strides[1], Strides[2]],
                               Padding = 'SAME',
                               ActivationFunction = 'Equal',
                               BatchNormalization = False,
                               Renormalization = False,
                               Training = Training,
                               Regularization = Regularization,
                               vname = vname + '_Conv_02c')
    x02 = Layers.concat(xs = [x1a, x1b, x1c], concat_type = 'Channel')
    if SE:
        x02 = SE_module(x = x02,
                        InputNode = [InputNode[0], InputNode[1], GrowthRate],
                        Act = Act,
                        vname = vname + '_SE02')
    return x02

def transition_cell(x,
                    Act = 'Relu',
                    InputNode = [64, 64, 32],
                    Strides = [1, 1, 1, 1],
                    Initializer = 'He',
                    Renormalization = False,
                    Regularization = False,
                    rmax = None,
                    dmax = None,
                    SE = True,
                    Training = True,
                    vname = '_Transition'):
    # Batch Normalization
    '''
    x_bn1 = Layers.batch_normalization(x = x,
                                       shape = InputNode[2],
                                       vname = vname + '_BN01',
                                       dim = [0, 1, 2],
                                       Renormalization = Renormalization,
                                       Training = Training,
                                       rmax = rmax,
                                       dmax = dmax)
    '''
    x_bn1 = Layers.group_normalization(x = x, G = 16,
                                       eps = 1e-5, vname = vanme + '_GN01')
    # Activation Function
    with tf.variable_scope(vname + '_Act01') as scope:
        x_act1 = AF.select_activation(Act)(x_bn1)

    x01 = Layers.convolution2d(x = x_act1,
                               FilterSize = [1, 1, InputNode[2], InputNode[2]],
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
                        InputNode = [InputNode[0], InputNode[1], InputNode[2]],
                        Act = Act,
                        vname = vname + '_SE01')
    p_max = Layers.pooling(x = x01,
                           ksize=[2, 2],
                           strides=[2, 2],
                           padding='SAME',
                           algorithm = 'Max')
    return p_max




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
                    Initializer = 'He' if Act in ['Relu', 'Gelu'] else 'Xavier',
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
