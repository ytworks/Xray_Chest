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
from LinearMotor import Visualizer as vs
from LinearMotor import Transfer as trans
from SimpleCells import *


def scratch_model(x, SIZE, CH, istraining, rmax, dmax, keep_probs, reuse=False):
    Initializer = 'He'
    Activation = 'Gelu'
    Regularization = False
    Renormalization = True
    SE = False
    GrowthRate = 8
    StemChannels = 32
    prob = 1.0
    GroupNum = 8
    GroupNorm = False
    # dense net
    # Stem
    # Batch Normalization
    stem_bn = Layers.batch_normalization(x=x,
                                         shape=CH,
                                         vname='STEM_TOP_BN01',
                                         dim=[0, 1, 2],
                                         Renormalization=Renormalization,
                                         Training=istraining,
                                         rmax=rmax,
                                         dmax=dmax)
    dense_stem = stem_cell(x=stem_bn,
                           InputNode=[SIZE, SIZE, CH],
                           Channels=StemChannels,
                           Initializer=Initializer,
                           vname='Stem',
                           regularization=Regularization,
                           Training=istraining)

    # Dense
    densenet_output = densenet(x=dense_stem,
                               root=stem_bn,
                               Act=Activation,
                               GrowthRate=GrowthRate,
                               InputNode=[SIZE / 4,
                                          SIZE / 4, StemChannels],
                               Strides=[1, 1, 1, 1],
                               Renormalization=Renormalization,
                               Regularization=Regularization,
                               rmax=rmax,
                               dmax=dmax,
                               SE=SE,
                               Training=istraining,
                               GroupNorm=GroupNorm,
                               GroupNum=GroupNum,
                               vname='DenseNet')

    y50 = densenet_output
    y51 = SE_module(x=y50,
                    InputNode=[SIZE / 64, SIZE / 64,
                               StemChannels + 12 + GrowthRate * 98],
                    Act=Activation,
                    Rate=0.5,
                    vname='TOP_SE')

    y61 = Layers.pooling(x=y51,
                         ksize=[SIZE / 64, SIZE / 64],
                         strides=[SIZE / 64, SIZE / 64],
                         padding='SAME',
                         algorithm='Avg')

    # reshape
    y71 = Layers.reshape_tensor(
        x=y61, shape=[StemChannels + 12 + GrowthRate * 98])
    y71_d = Layers.dropout(x=y71,
                           keep_probs=keep_probs,
                           training_prob=prob,
                           vname='Dropout')
    # fnn
    y72 = Outputs.output(x=y71_d,
                         InputSize=StemChannels + 12 + GrowthRate * 98,
                         OutputSize=15,
                         Initializer='Xavier',
                         BatchNormalization=False,
                         Regularization=True,
                         vname='Output_z')
    z = y72
    logit = tf.sigmoid(z)
    return z, logit, y51


def pretrain_model(x, rmax, dmax, keep_probs, reuse=False, is_train=True):
    p = trans.Transfer(x, 'densenet121', pooling=None, vname='transfer_Weight_Regularization',
                       trainable=True)
    # p.get_keys()
    y51 = p.get_output_tensor()
    y_3rd = p['conv4_block24_concat']
    y_3rd_tr = modified_transfer_layer(x=y_3rd,
                                       Act='Relu',
                                       InputNode=[14, 14, 1024],
                                       Strides=[1, 1, 1, 1],
                                       Initializer='He',
                                       Renormalization=True,
                                       Regularization=True,
                                       rmax=rmax,
                                       dmax=dmax,
                                       Training=is_train,
                                       vname='MODIFIED_Transition')
    y_3rd_dense = modified_dense_cell(x=y_3rd_tr,
                                      Num=16,
                                      Act='Relu',
                                      InputNode=[14, 14, 512],
                                      Initializer='He',
                                      GrowthRate=32,
                                      Strides=[1, 1, 1, 1],
                                      Renormalization=True,
                                      Regularization=False,
                                      rmax=rmax,
                                      dmax=dmax,
                                      Training=is_train,
                                      vname='MODIFIED_dense')
    y_3rd_bn = Layers.batch_normalization(x=y_3rd_dense,
                                       shape=1024,
                                       vname='BRANCH_BN01',
                                       dim=[0, 1, 2],
                                       Renormalization=True,
                                       Training=is_train,
                                       rmax=rmax,
                                       dmax=dmax)
    with tf.variable_scope('BRANCH_Act01') as scope:
        y_3rd_act = AF.select_activation('Relu')(y_3rd_bn)
    print(y_3rd_act)
    y52 = tf.image.resize_images(images=y51,
                                 size=[tf.constant(14, tf.int32), tf.constant(
                                     14, tf.int32)],
                                 method=tf.image.ResizeMethod.BICUBIC,
                                 align_corners=False)
    y53 = Layers.concat(xs=[y52, y_3rd_act], concat_type='Channel')
    m_size = 128
    tsl = Layers.convolution2d(x=y53,
                               FilterSize=[1, 1, 1024*2, 15 * m_size],
                               Initializer='He',
                               Strides=[1, 1],
                               Padding='SAME',
                               ActivationFunction='Equal',
                               BatchNormalization=False,
                               Renormalization=False,
                               Regularization=True,
                               Rmax=None,
                               Dmax=None,
                               Training=False,
                               vname='transfer_conv',
                               Is_log=False)
    cwp = Layers.class_wise_pooling(x=tsl, n_classes=15, m=m_size)
    print(cwp)
    z = Layers.spatial_pooling(
        x=cwp, k_train=20, k_test=20, alpha=0.7, is_train=is_train)
    logit = tf.sigmoid(z)
    return z, logit, cwp, p


def modified_transfer_layer(x,
                            Act='Relu',
                            InputNode=[64, 64, 32],
                            Strides=[1, 1, 1, 1],
                            Initializer='He',
                            Renormalization=False,
                            Regularization=False,
                            rmax=None,
                            dmax=None,
                            Training=True,
                            vname='_Transition'):
    x_bn1 = Layers.batch_normalization(x=x,
                                       shape=InputNode[2],
                                       vname=vname + '_BN01',
                                       dim=[0, 1, 2],
                                       Renormalization=Renormalization,
                                       Training=Training,
                                       rmax=rmax,
                                       dmax=dmax)
    with tf.variable_scope(vname + '_Act01') as scope:
        x_act1 = AF.select_activation(Act)(x_bn1)
    x01 = Layers.convolution2d(x=x_act1,
                               FilterSize=[1, 1, InputNode[2],
                                           InputNode[2] / 2],
                               Initializer=Initializer,
                               Strides=[Strides[1], Strides[2]],
                               Padding='SAME',
                               ActivationFunction='Equal',
                               BatchNormalization=False,
                               Renormalization=False,
                               Training=Training,
                               Regularization=Regularization,
                               vname=vname + '_Conv_01')
    return x01


def modified_conv_block(x,
                        Act='Relu',
                        InputNode=[64, 64, 32],
                        Initializer='He',
                        GrowthRate=32,
                        Strides=[1, 1, 1, 1],
                        Renormalization=False,
                        Regularization=False,
                        rmax=None,
                        dmax=None,
                        Training=True,
                        vname='_ConvBlock'):
    x_bn1 = Layers.batch_normalization(x=x,
                                       shape=InputNode[2],
                                       vname=vname + '_BN01',
                                       dim=[0, 1, 2],
                                       Renormalization=Renormalization,
                                       Training=Training,
                                       rmax=rmax,
                                       dmax=dmax)
    with tf.variable_scope(vname + '_Act01') as scope:
        x_act1 = AF.select_activation(Act)(x_bn1)
    x0 = Layers.convolution2d(x=x_act1,
                              FilterSize=[1, 1, InputNode[2], GrowthRate * 4],
                              Initializer=Initializer,
                              Strides=[Strides[1], Strides[2]],
                              Padding='SAME',
                              ActivationFunction='Equal',
                              BatchNormalization=False,
                              Renormalization=False,
                              Training=Training,
                              Regularization=Regularization,
                              vname=vname + '_Conv_01a')
    x_bn2 = Layers.batch_normalization(x=x0,
                                       shape=GrowthRate * 4,
                                       vname=vname + '_BN02',
                                       dim=[0, 1, 2],
                                       Renormalization=Renormalization,
                                       Training=Training,
                                       rmax=rmax,
                                       dmax=dmax)
    with tf.variable_scope(vname + '_Act02') as scope:
        x_act2 = AF.select_activation(Act)(x_bn2)
    x1 = Layers.dilated_convolution2d(x=x_act2,
                                      FilterSize=[
                                          3, 3, GrowthRate * 4, GrowthRate],
                                      Initializer=Initializer,
                                      Strides=[Strides[1], Strides[2]],
                                      Padding='SAME',
                                      ActivationFunction='Equal',
                                      BatchNormalization=False,
                                      Renormalization=False,
                                      Training=Training,
                                      Regularization=Regularization,
                                      vname=vname + '_Conv_02a')
    return x1


def modified_dense_cell(x,
                        Num=16,
                        Act='Relu',
                        GrowthRate=32,
                        InputNode=[64, 64, 32],
                        Initializer='He',
                        Strides=[1, 1, 1, 1],
                        Renormalization=False,
                        Regularization=False,
                        rmax=None,
                        dmax=None,
                        Training=True,
                        vname='_Dense'):
    outputs = [x]
    input = x
    for i in range(Num):
        outputs.append(modified_conv_block(x=input,
                                           Act=Act,
                                           GrowthRate=GrowthRate,
                                           InputNode=[InputNode[0], InputNode[1],
                                                      InputNode[2] + GrowthRate * i],
                                           Initializer=Initializer,
                                           Strides=[1, 1, 1, 1],
                                           Renormalization=Renormalization,
                                           Regularization=Regularization,
                                           rmax=rmax,
                                           dmax=dmax,
                                           Training=Training,
                                           vname=vname + '_ConvBlock' + str(i)))
        input = Layers.concat(xs=outputs, concat_type='Channel')
    return input
