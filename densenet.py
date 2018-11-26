import tensorflow as tf
import numpy as np
import sys
import os
import math
from datetime import datetime
from LinearMotor import ActivationFunctions as act
from LinearMotor import Loss
from LinearMotor import TrainOptimizers as opt
from LinearMotor import Utilities as ut
from LinearMotor import Layers
from LinearMotor import Outputs
from LinearMotor import Visualizer as vs
from LinearMotor import Transfer as trans
from LinearMotor import Variables
import cv2


def SE_module(x,
              InputNode,
              Act='Relu',
              Rate=0.5,
              vname='SE'):
    # Global Average Pooling
    x0 = Layers.pooling(x=x,
                        ksize=[InputNode[0], InputNode[1]],
                        strides=[InputNode[0], InputNode[1]],
                        padding='SAME',
                        algorithm='Avg')

    x1 = Layers.reshape_tensor(x=x0,
                               shape=[1 * 1 * InputNode[2]])

    x2 = Layers.fnn(x=x1,
                    InputSize=InputNode[2],
                    OutputSize=int(InputNode[2] * Rate),
                    Initializer='He' if Act in ['Relu', 'Gelu'] else 'Xavier',
                    ActivationFunction=Act,
                    MaxoutSize=3,
                    BatchNormalization=False,
                    Regularization=False,
                    vname=vname + '_FNN0')

    x3 = Layers.fnn(x=x2,
                    InputSize=int(InputNode[2] * Rate),
                    OutputSize=InputNode[2],
                    Initializer='Xavier_normal',
                    ActivationFunction='Sigmoid',
                    MaxoutSize=3,
                    BatchNormalization=False,
                    Regularization=False,
                    vname=vname + '_FNN1')

    x4 = Layers.reshape_tensor(x=x3,
                               shape=[1, 1, InputNode[2]])
    scale = x * x4
    return scale


def densenet121(x, is_train, rmax, dmax, ini, reuse=False, se=True, renorm=True, act_f='Relu'):
    with tf.variable_scope('Densenet101_Weight_Regularization', reuse=reuse):
        # TOP
        _, _, _, c = x.get_shape().as_list()
        y00 = Layers.batch_normalization(x=x,
                                         shape=c,
                                         vname='TOP_BN01',
                                         dim=[0, 1, 2],
                                         Training=is_train,
                                         Renormalization=renorm,
                                         Is_Fused=False,
                                         rmax=rmax,
                                         dmax=dmax)
        y01 = Layers.convolution2d(x=y00,
                                   FilterSize=[7, 7, c, 64],
                                   Initializer='He',
                                   Strides=[2, 2],
                                   Padding='SAME',
                                   ActivationFunction='Equal',
                                   BatchNormalization=False,
                                   Renormalization=False,
                                   Regularization=True,
                                   Rmax=None,
                                   Dmax=None,
                                   Training=is_train,
                                   vname='TOP_Conv01',
                                   Is_log=False,
                                   is_bias=True)
        y02 = Layers.batch_normalization(x=y01,
                                         shape=64,
                                         vname='TOP_BN02',
                                         dim=[0, 1, 2],
                                         Training=is_train,
                                         Renormalization=renorm,
                                         Is_Fused=False,
                                         rmax=rmax,
                                         dmax=dmax)
        with tf.variable_scope('TOP_Act01') as scope:
            y03 = act.select_activation(act_f)(y02)
        y04 = Layers.pooling(x=y03,
                             ksize=[3, 3],
                             strides=[2, 2],
                             padding='SAME',
                             algorithm='Max')
        # 6 + 12 + 24 + 6
        y11 = dense_block(x=y04,
                          blocks=6,
                          is_train=is_train,
                          rmax=rmax,
                          dmax=dmax,
                          vname='Dense01', renorm=renorm, act_f=act_f)
        y12 = transition_block(x=y11,
                               reduction=0.5,
                               is_train=is_train,
                               rmax=rmax,
                               dmax=dmax,
                               vname="Transition1", renorm=renorm, act_f=act_f)
        if se:
            _, h, w, c = y12.get_shape().as_list()
            y13 = SE_module(x=y12,
                            InputNode=[h, w, c],
                            Act=act_f,
                            Rate=0.5,
                            vname='SE1')
        else:
            y13 = y12
        y21 = dense_block(x=y13,
                          blocks=12,
                          is_train=is_train,
                          rmax=rmax,
                          dmax=dmax,
                          vname='Dense02', renorm=renorm, act_f=act_f)
        y22 = transition_block(x=y21,
                               reduction=0.5,
                               is_train=is_train,
                               rmax=rmax,
                               dmax=dmax,
                               vname="Transition2", renorm=renorm, act_f=act_f)
        if se:
            _, h, w, c = y22.get_shape().as_list()
            y23 = SE_module(x=y22,
                            InputNode=[h, w, c],
                            Act=act_f,
                            Rate=0.5,
                            vname='SE2')
        else:
            y23 = y22
        y31 = dense_block(x=y23,
                          blocks=24,
                          is_train=is_train,
                          rmax=rmax,
                          dmax=dmax,
                          vname='Dense03', renorm=renorm, act_f=act_f)
        y32 = transition_block(x=y31,
                               reduction=0.5,
                               is_train=is_train,
                               rmax=rmax,
                               dmax=dmax,
                               vname="Transition3", renorm=renorm, act_f=act_f)
        if se:
            _, h, w, c = y32.get_shape().as_list()
            y33 = SE_module(x=y32,
                            InputNode=[h, w, c],
                            Act=act_f,
                            Rate=0.5,
                            vname='SE3')
        else:
            y33 = y32
        y41 = dense_block(x=y33,
                          blocks=16,
                          is_train=is_train,
                          rmax=rmax,
                          dmax=dmax,
                          vname='Dense04', renorm=renorm, act_f=act_f)
        _, _, _, c = y41.get_shape().as_list()
        y42 = Layers.batch_normalization(x=y41,
                                         shape=c,
                                         vname='LAST_BN01',
                                         dim=[0, 1, 2],
                                         Training=is_train,
                                         Renormalization=renorm,
                                         Is_Fused=False,
                                         rmax=rmax,
                                         dmax=dmax)
        tsl = Layers.convolution2d(x=y42,
                                   FilterSize=[1, 1, 1024, 15 *
                                               ini.getint('DLParams', 'wc_m')],
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
        tsl = Layers.batch_normalization(x=tsl,
                                         shape=c,
                                         vname='LAST_BN02',
                                         dim=[0, 1, 2],
                                         Training=is_train,
                                         Renormalization=renorm,
                                         Is_Fused=False,
                                         rmax=rmax,
                                         dmax=dmax)
        y51 = Layers.class_wise_pooling(x=tsl,
                                        n_classes=15,
                                        m=ini.getint(
                                            'DLParams', 'wc_m'))
        y61 = Layers.spatial_pooling(x=y51,
                                     k_train=ini.getint(
                                         'DLParams', 'wc_k'),
                                     k_test=ini.getint(
                                         'DLParams', 'wc_k'),
                                     alpha=ini.getfloat(
                                         'DLParams', 'wc_alpha'),
                                     is_train=is_train)
        y = y61
        logit = tf.sigmoid(y)
        return y, logit, y51


def dense_block(x, blocks, is_train, rmax, dmax, vname, renorm=True, act_f='Relu'):
    for i in range(blocks):
        x = conv_block(x, 32, is_train, rmax, dmax,
                       vname=vname + '_block' + str(i + 1),
                       renorm=renorm,
                       act_f=act_f)
    return x


def conv_block(x, growth_rate, is_train, rmax, dmax, vname, renorm=True, act_f='Relu'):
    _, _, _, c = x.get_shape().as_list()
    x01 = Layers.batch_normalization(x=x,
                                     shape=c,
                                     vname=vname + '_BN01',
                                     dim=[0, 1, 2],
                                     Training=is_train,
                                     Renormalization=renorm,
                                     Is_Fused=False,
                                     rmax=rmax,
                                     dmax=dmax)
    with tf.variable_scope(vname + '_Act01') as scope:
        x02 = act.select_activation(act_f)(x01)
    x03 = Layers.convolution2d(x=x02,
                               FilterSize=[1, 1, c, 4 * growth_rate],
                               Initializer='He',
                               Strides=[1, 1],
                               Padding='SAME',
                               ActivationFunction='Equal',
                               BatchNormalization=False,
                               Renormalization=False,
                               Regularization=True,
                               Rmax=None,
                               Dmax=None,
                               Training=is_train,
                               vname=vname + '_Conv01',
                               Is_log=False)
    x05 = Layers.batch_normalization(x=x03,
                                     shape=4 * growth_rate,
                                     vname=vname + '_BN02',
                                     dim=[0, 1, 2],
                                     Training=is_train,
                                     Renormalization=renorm,
                                     Is_Fused=False,
                                     rmax=rmax,
                                     dmax=dmax)
    with tf.variable_scope(vname + '_Act02') as scope:
        x06 = act.select_activation(act_f)(x05)
    x07 = Layers.convolution2d(x=x06,
                               FilterSize=[3, 3, 4 * growth_rate, growth_rate],
                               Initializer='He',
                               Strides=[1, 1],
                               Padding='SAME',
                               ActivationFunction='Equal',
                               BatchNormalization=False,
                               Renormalization=False,
                               Regularization=True,
                               Rmax=None,
                               Dmax=None,
                               Training=is_train,
                               vname=vname + '_Conv02',
                               Is_log=False)
    x07 = sfcm(x1=x, x2=x07, is_train=is_train, rmax=rmax, dmax=dmax, renorm=renorm, act_f=act_f, vname=vname+'_SFCM')
    x08 = Layers.concat(xs=[x, x07], concat_type='Channel')
    return x08


def sfcm(x1, x2, is_train, rmax, dmax, renorm, act_f, vname):
    _, h, w, c = x2.get_shape().as_list()
    gm = Layers.convolution2d(x=x2,
                               FilterSize=[1, 1, c, 1],
                               Initializer='Xavier_normal',
                               Strides=[1, 1],
                               Padding='SAME',
                               ActivationFunction='Equal',
                               BatchNormalization=False,
                               Renormalization=False,
                               Regularization=True,
                               Rmax=None,
                               Dmax=None,
                               Training=is_train,
                               vname=vname + '_SFCM_Conv01',
                               Is_log=False)
    gms = tf.nn.softmax(gm)
    _, h, w, c = x1.get_shape().as_list()
    gms = tf.broadcast_to(gms, [-1, h, w, c])
    print(gms)
    gmsp = x1 * gms
    print('GMSP', gmsp)
    wx = Variables.bias_variable(shape = [1], initial_value = 1.0, vname = vname + '_SCALE')
    return x1 + wx * gmsp


def transition_block(x, reduction, is_train, rmax, dmax, vname, renorm=True, act_f='Relu'):
    _, _, _, c = x.get_shape().as_list()
    x01 = Layers.batch_normalization(x=x,
                                     shape=c,
                                     vname=vname + '_BN01',
                                     dim=[0, 1, 2],
                                     Training=is_train,
                                     Renormalization=renorm,
                                     Is_Fused=False,
                                     rmax=rmax,
                                     dmax=dmax)
    with tf.variable_scope(vname + '_Act01') as scope:
        x02 = act.select_activation(act_f)(x01)
    x03 = Layers.convolution2d(x=x02,
                               FilterSize=[1, 1, c, c * reduction],
                               Initializer='He',
                               Strides=[1, 1],
                               Padding='SAME',
                               ActivationFunction='Equal',
                               BatchNormalization=False,
                               Renormalization=False,
                               Regularization=True,
                               Rmax=None,
                               Dmax=None,
                               Training=is_train,
                               vname=vname + '_Conv01',
                               Is_log=False)
    x04 = Layers.pooling(x=x03,
                         ksize=[2, 2],
                         strides=[2, 2],
                         padding='SAME',
                         algorithm='Max')
    return x04


def attention_block(x, is_train, rmax, dmax, vname, renorm=True, act_f='Relu'):
    _, h, w, c = x.get_shape().as_list()
    x01 = res_block(x=x, is_train=is_train, rmax=rmax, dmax=dmax,
                    vname=vname + '_r01', renorm=True, act_f='Relu')
    s01 = res_block(x=x01, is_train=is_train, rmax=rmax, dmax=dmax,
                    vname=vname + '_s01', renorm=True, act_f='Relu')
    s02 = res_block(x=s01, is_train=is_train, rmax=rmax, dmax=dmax,
                    vname=vname + '_s02', renorm=True, act_f='Relu')
    t01 = Layers.pooling(x=x01,
                         ksize=[2, 2],
                         strides=[2, 2],
                         padding='SAME',
                         algorithm='Max')
    t02 = res_block(x=t01, is_train=is_train, rmax=rmax, dmax=dmax,
                    vname=vname + '_t02', renorm=True, act_f='Relu')
    t03 = Layers.pooling(x=t02,
                         ksize=[2, 2],
                         strides=[2, 2],
                         padding='SAME',
                         algorithm='Max')
    t04 = res_block(x=t03, is_train=is_train, rmax=rmax, dmax=dmax,
                    vname=vname + '_t03', renorm=True, act_f='Relu')
    t05 = res_block(x=t04, is_train=is_train, rmax=rmax, dmax=dmax,
                    vname=vname + '_t04', renorm=True, act_f='Relu')
    t06 = tf.image.resize_images(images=t05,
                                 size=[tf.constant(
                                     int(h / 2), tf.int32), tf.constant(int(w / 2), tf.int32)],
                                 method=tf.image.ResizeMethod.BICUBIC,
                                 align_corners=False)
    t07 = res_block(x=t06, is_train=is_train, rmax=rmax, dmax=dmax,
                    vname=vname + '_t06', renorm=True, act_f='Relu')
    t08 = tf.image.resize_images(images=t07,
                                 size=[tf.constant(h, tf.int32),
                                       tf.constant(w, tf.int32)],
                                 method=tf.image.ResizeMethod.BICUBIC,
                                 align_corners=False)
    t10 = Layers.convolution2d(x=t08,
                               FilterSize=[1, 1, c, c],
                               Initializer='He',
                               Strides=[1, 1],
                               Padding='SAME',
                               ActivationFunction='Equal',
                               BatchNormalization=False,
                               Renormalization=False,
                               Regularization=True,
                               Rmax=None,
                               Dmax=None,
                               Training=is_train,
                               vname=vname + '_Conv01',
                               Is_log=False)
    t11 = Layers.convolution2d(x=t10,
                               FilterSize=[1, 1, c, c],
                               Initializer='He',
                               Strides=[1, 1],
                               Padding='SAME',
                               ActivationFunction='Equal',
                               BatchNormalization=False,
                               Renormalization=False,
                               Regularization=True,
                               Rmax=None,
                               Dmax=None,
                               Training=is_train,
                               vname=vname + '_Conv02',
                               Is_log=False)
    t12 = tf.sigmoid(t11)
    m01 = s02 + s02 * t12
    m02 = res_block(x=m01, is_train=is_train, rmax=rmax, dmax=dmax,
                    vname=vname + '_m02', renorm=True, act_f='Relu')
    return m02


def res_block(x, is_train, rmax, dmax, vname, renorm=True, act_f='Relu'):
    _, _, _, c = x.get_shape().as_list()
    x01 = Layers.batch_normalization(x=x,
                                     shape=c,
                                     vname=vname + '_BN01',
                                     dim=[0, 1, 2],
                                     Training=is_train,
                                     Renormalization=renorm,
                                     Is_Fused=False,
                                     rmax=rmax,
                                     dmax=dmax)
    with tf.variable_scope(vname + '_Act01') as scope:
        x02 = act.select_activation(act_f)(x01)
    x03 = Layers.convolution2d(x=x02,
                               FilterSize=[3, 3, c, c],
                               Initializer='He',
                               Strides=[1, 1],
                               Padding='SAME',
                               ActivationFunction='Equal',
                               BatchNormalization=False,
                               Renormalization=False,
                               Regularization=True,
                               Rmax=None,
                               Dmax=None,
                               Training=is_train,
                               vname=vname + '_Conv01',
                               Is_log=False)
    x04 = Layers.batch_normalization(x=x03,
                                     shape=c,
                                     vname=vname + '_BN02',
                                     dim=[0, 1, 2],
                                     Training=is_train,
                                     Renormalization=renorm,
                                     Is_Fused=False,
                                     rmax=rmax,
                                     dmax=dmax)
    with tf.variable_scope(vname + '_Act02') as scope:
        x05 = act.select_activation(act_f)(x04)
    x06 = Layers.convolution2d(x=x05,
                               FilterSize=[3, 3, c, c],
                               Initializer='He',
                               Strides=[1, 1],
                               Padding='SAME',
                               ActivationFunction='Equal',
                               BatchNormalization=False,
                               Renormalization=False,
                               Regularization=True,
                               Rmax=None,
                               Dmax=None,
                               Training=is_train,
                               vname=vname + '_Conv02',
                               Is_log=False)
    return x + x06
