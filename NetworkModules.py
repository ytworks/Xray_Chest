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


def synplectic_scratch_model(x, SIZE, CH, istraining, rmax, dmax, keep_probs):
    Initializer = 'He'
    Activation = 'Gelu'
    Regularization = False
    Renormalization = True
    SE = False
    BottleNeck = 2
    StemChannels = 32
    GrowthRate = 8
    prob = 1.0
    GroupNum = 8
    GroupNorm = False
    Nums = [6, 12, 48, 32]
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
    densenet_output = synplectic_densenet_template(x=dense_stem,
                                                   Nums=Nums,
                                                   Act=Activation,
                                                   BottleNeck=BottleNeck,
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
                               StemChannels + GrowthRate * (sum(Nums))],
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
        x=y61, shape=[StemChannels + GrowthRate * (sum(Nums))])
    y71_d = Layers.dropout(x=y71,
                           keep_probs=keep_probs,
                           training_prob=prob,
                           vname='Dropout')
    # fnn
    y72 = Outputs.output(x=y71_d,
                         InputSize=StemChannels + GrowthRate* (sum(Nums)),
                         OutputSize=15,
                         Initializer='Xavier',
                         BatchNormalization=False,
                         Regularization=True,
                         vname='Output_z')
    z = y72
    logit = tf.sigmoid(z)
    return z, logit, y51


def scratch_model(x, SIZE, CH, istraining, rmax, dmax, keep_probs):
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


def pretrain_model(x):
    p = trans.Transfer(x, 'densenet201', pooling=None, vname='Transfer',
                       trainable=True)
    y51 = p.get_output_tensor()
    y61 = Layers.pooling(x=y51,
                         ksize=[7, 7],
                         strides=[7, 7],
                         padding='SAME',
                         algorithm='Avg')

    # reshape
    y71 = Layers.reshape_tensor(x=y61, shape=[1 * 1 * 1920])
    # fnn
    y72 = Outputs.output(x=y71,
                         InputSize=1920,
                         OutputSize=15,
                         Initializer='Xavier_normal',
                         BatchNormalization=False,
                         Regularization=True,
                         vname='Output_z',
                         Is_bias=True)
    z = y72
    logit = tf.sigmoid(z)
    return z, logit, y51, p
