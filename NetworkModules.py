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


def pretrain_model(x, config, reuse=False, is_train=True):
    k_size = config.getint('DLParams', 'wc_k')
    alpha = config.getfloat('DLParams', 'wc_alpha')
    p = trans.Transfer(x, 'densenet121', pooling=None, vname='transfer_Weight_Regularization',
                       trainable=True)
    y51 = p.get_output_tensor()
    pool4 = p['pool4_conv']
    pool3 = p['pool3_conv']
    pool2 = p['pool2_conv']
    pool3 = tf.image.resize_images(images=pool3,
                                   size=[tf.constant(14, tf.int32), tf.constant(14, tf.int32)],
                                   method=tf.image.ResizeMethod.BICUBIC,
                                   align_corners=False)
    pool2 = tf.image.resize_images(images=pool2,
                                   size=[tf.constant(14, tf.int32), tf.constant(14, tf.int32)],
                                   method=tf.image.ResizeMethod.BICUBIC,
                                   align_corners=False)
    y51 = tf.image.resize_images(images=y51,
                                   size=[tf.constant(14, tf.int32), tf.constant(14, tf.int32)],
                                   method=tf.image.ResizeMethod.BICUBIC,
                                   align_corners=False)
    feature = Layers.concat(xs=[y51, pool2, pool3, pool4], concat_type='Channel')
    m_size = 128
    tsl = Layers.convolution2d(x=feature,
                               FilterSize=[1, 1, 1024+512+256+128, 15 * m_size],
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
    tsl = SE_module(x=tsl, InputNode=[14,14,15*m_size], Act='Relu', Rate=0.5, vname='SE')
    cwp = Layers.class_wise_pooling(x=tsl, n_classes=15, m=m_size)
    print(cwp)
    z = Layers.spatial_pooling_dif(x=cwp, k_train=k_size, k_test=k_size, alpha=alpha, is_train=is_train)
    logit = tf.sigmoid(z)
    return z, logit, cwp, p
