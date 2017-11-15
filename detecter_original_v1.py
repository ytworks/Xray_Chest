#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import sys
import os
import math
from LinearMotor import Core2
from LinearMotor import ActivationFunctions as AF
from LinearMotor import Layers
from LinearMotor import Outputs
from LinearMotor import TrainOptimizers as TO
from LinearMotor import Utilities as UT
from LinearMotor import Loss
from LinearMotor import Cells
from LinearMotor.Cells2 import inception_res_cell
from logging import getLogger, StreamHandler
logger = getLogger(__name__)
sh = StreamHandler()
logger.addHandler(sh)
logger.setLevel(10)

class Detecter(Core2.Core):
    def __init__(self,
                 output_type,
                 epoch = 100, batch = 32, log = 10,
                 optimizer_type = 'Adam',
                 learning_rate = 0.0001,
                 dynamic_learning_rate = 0.000001,
                 beta1 = 0.9, beta2 = 0.999,
                 regularization = 0.0,
                 regularization_type = 'L2',
                 checkpoint = './Storages/Core.ckpt',
                 init = True,
                 size = 256):
        super(Detecter, self).__init__(output_type = output_type,
                                       epoch = epoch,
                                       batch = batch,
                                       log = log,
                                       optimizer_type = optimizer_type,
                                       learning_rate = learning_rate,
                                       dynamic_learning_rate = dynamic_learning_rate,
                                       beta1 = beta1,
                                       beta2 = beta2,
                                       regularization = regularization,
                                       regularization_type = regularization_type,
                                       checkpoint = checkpoint,
                                       init = init
                                       )
        self.SIZE = size

    def construct(self):
        # セッションの定義
        self.sess = tf.InteractiveSession()
        logger.debug("01: TF session Start")
        # 入出力の定義
        self.io_def()
        logger.debug("02: TF I/O definition done")
        # ネットワークの構成
        self.network()
        logger.debug("03: TF network construction done")
        # 誤差関数の定義
        self.loss()
        logger.debug("04: TF Loss definition done")
        # 学習
        self.training()
        logger.debug("05: TF Training operation done")
        # 精度の定義
        #self.get_accuracy()
        #print(self.accuracy)
        # チェックポイントの呼び出し
        #self.saver = tf.train.Saver()
        #self.restore()

    def io_def(self):
        self.CH = 1
        self.x = tf.placeholder("float", shape=[None, self.SIZE, self.SIZE, self.CH], name = "Input")
        self.y_ = tf.placeholder("float", shape=[None, 2], name = "Label_Judgement")
        self.z_ = tf.placeholder("float", shape=[None, 14], name = "Label_Diagnosis")
        self.keep_probs = []

    def network(self):
        Channels = 10
        Initializer = 'He'
        Parallels = 9
        Activation = 'PRelu'
        Regularization = False
        prob = 1.0
        self.y11 = inception_res_cell(x = self.x,
                                           Act = Activation,
                                           InputNode = [self.SIZE, self.SIZE, self.CH],
                                           Channels = [Channels, Channels],
                                           Strides0 = [1, 1, 1, 1],
                                           Strides1 = [1, 1, 1, 1],
                                           Initializer = Initializer,
                                           Regularization = Regularization,
                                           vname = 'Res0')
        self.y12 = Layers.pooling(x = self.y11, ksize=[2, 2], strides=[2, 2],
                                  padding='SAME', algorithm = 'Max')

        self.y21 = inception_res_cell(x = self.y12,
                                           Act = Activation,
                                           InputNode = [self.SIZE / 2, self.SIZE / 2, Channels * Parallels],
                                           Channels = [Channels * 2, Channels * 2],
                                           Strides0 = [1, 1, 1, 1],
                                           Strides1 = [1, 1, 1, 1],
                                           Initializer = Initializer,
                                           Regularization = Regularization,
                                           vname = 'Res1')
        self.y21 = Layers.pooling(x = self.y21, ksize=[2, 2], strides=[2, 2],
                                  padding='SAME', algorithm = 'Max')


        self.y31 = inception_res_cell(x = self.y21,
                                           Act = Activation,
                                           InputNode = [self.SIZE / 4, self.SIZE / 4, Channels * 2 * Parallels],
                                           Channels = [Channels * 4, Channels * 4],
                                           Strides0 = [1, 1, 1, 1],
                                           Strides1 = [1, 1, 1, 1],
                                           Initializer = Initializer,
                                           Regularization = Regularization,
                                           vname = 'Res3')
        self.y31 = Layers.pooling(x = self.y31, ksize=[2, 2], strides=[2, 2],
                                  padding='SAME', algorithm = 'Max')

        self.y41 = inception_res_cell(x = self.y31,
                                           Act = Activation,
                                           InputNode = [self.SIZE / 8, self.SIZE / 8, Channels * 4 * Parallels],
                                           Channels = [Channels * 8, Channels * 8],
                                           Strides0 = [1, 1, 1, 1],
                                           Strides1 = [1, 1, 1, 1],
                                           Initializer = Initializer,
                                           Regularization = Regularization,
                                           vname = 'Res4')
        self.y41 = Layers.pooling(x = self.y41, ksize=[2, 2], strides=[2, 2],
                                  padding='SAME', algorithm = 'Max')


        self.y51 = inception_res_cell(x = self.y41,
                                           Act = Activation,
                                           InputNode = [self.SIZE / 16, self.SIZE / 16, Channels * 8 * Parallels],
                                           Channels = [Channels * 16, Channels * 16],
                                           Strides0 = [1, 1, 1, 1],
                                           Strides1 = [1, 1, 1, 1],
                                           Initializer = Initializer,
                                           Regularization = Regularization,
                                           vname = 'Res5')
        # Sparse Target Layer
        self.y51 = Layers.pooling(x = self.y51, ksize=[2, 2], strides=[2, 2],
                                  padding='SAME', algorithm = 'Max')
        # Dropout Layer
        self.y52 = Layers.dropout(x = self.y51, keep_probs = self.keep_probs, training_prob = prob, vname = 'V5')

        self.y61 = Layers.pooling(x = self.y52,
                                  ksize=[self.SIZE / 16, self.SIZE / 16],
                                  strides=[self.SIZE / 16, self.SIZE / 16],
                                  padding='SAME',
                                  algorithm = 'Avg')


        # reshape
        self.y71 = Layers.reshape_tensor(x = self.y61, shape = [1 * 1 * Channels * 16 * Parallels])
        # fnn
        self.y72 = Outputs.output(x = self.y71,
                                  InputSize = Channels * 16 * Parallels,
                                  OutputSize = 2,
                                  Initializer = 'Xavier',
                                  BatchNormalization = False,
                                  Regularization = Regularization,
                                  vname = 'Output_y')
        self.y = self.y72
        self.y73 = Outputs.output(x = self.y71,
                                  InputSize = Channels * 16 * Parallels,
                                  OutputSize = 14,
                                  Initializer = 'Xavier',
                                  BatchNormalization = False,
                                  Regularization = Regularization,
                                  vname = 'Output_z')
        self.z = self.y73


    def loss(self):
        self.loss_function = Loss.loss_func(y = self.y,
                                            y_ = self.y_,
                                            regularization = self.regularization,
                                            regularization_type = self.regularization_type,
                                            output_type = self.output_type)
        self.loss_function += Loss.loss_func(y = self.z,
                                             y_ = self.z_,
                                             regularization = 0.0,
                                             regularization_type = self.regularization_type,
                                             output_type = 'classified-sigmoid')
        # For Gear Mode (TBD)
        #+ tf.reduce_mean(tf.abs(self.y51)) * self.GearLevel
