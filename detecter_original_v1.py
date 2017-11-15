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
        #self.network()
        # 誤差関数の定義
        #self.loss()
        # 学習
        #self.training()
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
