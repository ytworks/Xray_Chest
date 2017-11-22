#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import cv2
import sys
import os
import math
from DICOMReader.DICOMReader import dicom_to_np
from tqdm import tqdm
from datetime import datetime
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

'''
ToDo
    1. weighted lossの実装
    2. CAMの実装


'''


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
        self.accuracy_y = UT.correct_rate(self.y, self.y_)
        self.accuracy_z = tf.sqrt(tf.reduce_mean(tf.multiply(tf.sigmoid(self.z) -self.z_, tf.sigmoid(self.z) -self.z_)))
        logger.debug("06: TF Accuracy measure definition done")
        # チェックポイントの呼び出し
        self.saver = tf.train.Saver()
        self.restore()
        logger.debug("07: TF Model file definition done")




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
                                  OutputSize = 14,
                                  Initializer = 'Xavier',
                                  BatchNormalization = False,
                                  Regularization = Regularization,
                                  vname = 'Output_z')
        self.z0 = Layers.concat([self.y72, self.y71], concat_type = 'Vector')
        self.y73 = Outputs.output(x = self.z0,
                                  InputSize = Channels * 16 * Parallels + 14,
                                  OutputSize = 2,
                                  Initializer = 'Xavier',
                                  BatchNormalization = False,
                                  Regularization = Regularization,
                                  vname = 'Output_y')
        self.y = self.y73
        self.z = self.y72


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

    # 入出力ベクトルの配置
    def make_feed_dict(self, prob, batch):
        feed_dict = {}
        feed_dict.setdefault(self.x, batch[0])
        feed_dict.setdefault(self.y_, batch[1])
        feed_dict.setdefault(self.z_, batch[2])
        feed_dict.setdefault(self.learning_rate, self.learning_rate_value)
        #feed_dict.setdefault(self.GearLevel, self.GearLevelValue)
        i = 0
        for keep_prob in self.keep_probs:
            if prob:
                feed_dict.setdefault(keep_prob['var'], 1.0)
            else:
                feed_dict.setdefault(keep_prob['var'], keep_prob['prob'])
            i += 1
        return feed_dict


    def learning(self, data, save_at_log = False, validation_batch_num = 40):
        for i in tqdm(range(self.epoch)):

            batch = data.train.next_batch(self.batch)

            # 途中経過のチェック
            if i%self.log == 0:
                # Train
                feed_dict = self.make_feed_dict(prob = True, batch = batch)
                train_accuracy_y = self.accuracy_y.eval(feed_dict=feed_dict)
                train_accuracy_z = self.accuracy_z.eval(feed_dict=feed_dict)
                losses = self.loss_function.eval(feed_dict=feed_dict)
                # Test
                val_accuracy_y, val_accuracy_z, val_losses = [], [], []
                for num in range(validation_batch_num):
                    validation_batch = data.test.next_batch(self.batch, augment = False)
                    feed_dict_val = self.make_feed_dict(prob = False, batch = validation_batch)
                    val_accuracy_y.append(self.accuracy_y.eval(feed_dict=feed_dict_val) * float(self.batch))
                    val_accuracy_z.append(self.accuracy_z.eval(feed_dict=feed_dict_val) * float(self.batch))
                    val_losses.append(self.loss_function.eval(feed_dict=feed_dict_val) * float(self.batch))
                val_accuracy_y = np.mean(val_accuracy_y) / float(self.batch)
                val_accuracy_z = np.mean(val_accuracy_z) / float(self.batch)
                val_losses = np.mean(val_losses) / float(self.batch)
                # Output
                logger.debug("step %d train acc judgement %g train acc diagnosis %g Loss train %g validation acc judgement %g validation acc diagnosis %g Loss validation %g" % (i,train_accuracy_y,train_accuracy_z,losses,val_accuracy_y,val_accuracy_z,val_losses))
                if save_at_log:
                    self.save_checkpoint()
            # 学習
            feed_dict = self.make_feed_dict(prob = False, batch = batch)
            if self.DP and i != 0:
                self.dynamic_learning_rate(feed_dict)
            self.train_op.run(feed_dict=feed_dict)
        self.save_checkpoint()


    def get_output_weights(self, feed_dict):
        tvar = self.get_trainable_var()
        output_vars = []
        for var in tvar:
            if var.name.find('Output_z') >= 0:
                output_vars.append(var)
        weights = self.sess.run(output_vars, feed_dict = feed_dict)
        return weights


    def get_roi_map_base(self, feed_dict):
        return self.sess.run([self.y51], feed_dict = feed_dict)


    # 予測器
    def prediction(self, data, roi = False, label_def = None, save_dir = None,
                   filename = None, path = None):
        # Make feed dict for prediction
        feed_dict = {self.x : data}
        for keep_prob in self.keep_probs:
            feed_dict.setdefault(keep_prob['var'], 1.0)

        if not roi:
            result = self.sess.run(tf.nn.softmax(self.y), feed_dict = feed_dict)
            return result, None
        else:
            result = self.sess.run(tf.nn.softmax(self.y), feed_dict = feed_dict)
            weights = self.get_output_weights(feed_dict = feed_dict)
            roi_base = self.get_roi_map_base(feed_dict = feed_dict)
            print(weights[0].shape)
            print(roi_base[0].shape)
            self.make_roi(weights = weights[0],
                          roi_base = roi_base[0],
                          save_dir = save_dir,
                          filename = filename,
                          label_def = label_def,
                          path = path)

            return result, None

    def make_roi(self, weights, roi_base, save_dir, filename, label_def, path):
        img, bits = dicom_to_np(path)
        img = img / bits * 255
        img = img.astype(np.uint8)
        img = cv2.resize(img, (self.SIZE, self.SIZE), interpolation = cv2.INTER_AREA)
        img = np.stack((img, img, img), axis = -1)
        for x, finding in enumerate(label_def):
            logger.debug('Make ROI: %s'%finding)
            images = np.zeros((roi_base.shape[1], roi_base.shape[2], 3))
            for channel in range(roi_base.shape[2]):
                c = roi_base[0, :, :, channel]
                image = np.stack((c, c, c), axis = -1)
                images += image * weights[channel][x]
            images = 255.0 * (images - np.min(images)) / (np.max(images) - np.min(images))
            images = cv2.applyColorMap(images.astype(np.uint8), cv2.COLORMAP_JET)
            images = cv2.resize(images, (self.SIZE, self.SIZE))
            print(images.shape, img.shape)
            roi_img = cv2.addWeighted(img, 0.7, images, 0.3, 1.0)
            cv2.imwrite(save_dir + '/' + str(filename[0]) + '_' + str(finding) + '.png', roi_img)
