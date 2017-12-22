#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import cv2
import sys
import os
import math
import time
from DICOMReader.DICOMReader import dicom_to_np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from datetime import datetime
from LinearMotor import Core2
from LinearMotor import ActivationFunctions as AF
from LinearMotor import Layers
from LinearMotor import Outputs
from LinearMotor import TrainOptimizers as TO
from LinearMotor import Utilities as UT
from LinearMotor import Loss
from Cells import inception_res_cell
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
                 size = 256,
                 l1_norm = 0.1):
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
        self.l1_norm = l1_norm

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
        #self.accuracy_y = UT.correct_rate(self.y, self.y_)
        if self.output_type.find('hinge') >= 0:
            #self.accuracy_z = tf.sqrt(tf.reduce_mean(tf.multiply(self.z - self.z_, self.z - self.z_)))
            self.accuracy_z = tf.reduce_mean(tf.keras.metrics.cosine_proximity(self.z_, self.z))
        else:
            #self.accuracy_z = tf.sqrt(tf.reduce_mean(tf.multiply(tf.sigmoid(self.z) -self.z_, tf.sigmoid(self.z) -self.z_)))
            self.accuracy_z = tf.reduce_mean(tf.keras.metrics.cosine_proximity(self.z_, tf.sigmoid(self.z)))
        logger.debug("06: TF Accuracy measure definition done")
        # チェックポイントの呼び出し
        self.saver = tf.train.Saver()
        self.restore()
        logger.debug("07: TF Model file definition done")




    def io_def(self):
        self.CH = 1
        self.x = tf.placeholder("float", shape=[None, self.SIZE, self.SIZE, self.CH], name = "Input")
        #self.y_ = tf.placeholder("float", shape=[None, 2], name = "Label_Judgement")
        self.z_ = tf.placeholder("float", shape=[None, 14], name = "Label_Diagnosis")
        self.keep_probs = []

    def network(self):
        Channels = 16
        Initializer = 'He'
        Parallels = 9
        Activation = 'Relu'
        Regularization = True
        SE = True
        prob = 1.0
        self.x0 = Layers.batch_normalization(x = self.x, shape = [0, 1, 2], vname = 'First_BN',
                                             Renormalization = True, Training = self.istraining)
        self.y11 = inception_res_cell(x = self.x0,
                                           Act = Activation,
                                           InputNode = [self.SIZE, self.SIZE, self.CH],
                                           Channels = [Channels, Channels],
                                           Strides0 = [1, 1, 1, 1],
                                           Strides1 = [1, 1, 1, 1],
                                           Initializer = Initializer,
                                           Regularization = Regularization,
                                           vname = 'Res0',
                                           Training = self.istraining,
                                           SE = SE,
                                           STEM = False)
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
                                           Training = self.istraining,
                                           SE = SE,
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
                                           Training = self.istraining,
                                           SE = SE,
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
                                           Training = self.istraining,
                                           SE = SE,
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
                                           Training = self.istraining,
                                           SE = SE,
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
                                  Initializer = 'Xavier_normal',
                                  BatchNormalization = False,
                                  Regularization = True,
                                  vname = 'Output_z')
        '''
        self.z0 = Layers.concat([self.y72, self.y71], concat_type = 'Vector')
        self.y73 = Outputs.output(x = self.z0,
                                  InputSize = Channels * 16 * Parallels + 14,
                                  OutputSize = 2,
                                  Initializer = 'Xavier_normal',
                                  BatchNormalization = False,
                                  Regularization = True,
                                  vname = 'Output_y')
        self.y = self.y73
        '''
        self.z = self.y72


    def loss(self):
        diag_output_type = self.output_type if self.output_type.find('hinge') >= 0 else 'classified-sigmoid'

        self.loss_function = Loss.loss_func(y = self.z,
                                             y_ = self.z_,
                                             regularization = self.regularization,
                                             regularization_type = self.regularization_type,
                                             output_type = self.output_type)
        '''
        self.loss_function += Loss.loss_func(y = self.z,
                                             y_ = self.z_,
                                             regularization = 0.0,
                                             regularization_type = self.regularization_type,
                                             output_type = 'classified-cosine_proximity')
        '''
        '''
        self.loss_function += Loss.loss_func(y = self.y,
                                            y_ = self.y_,
                                            regularization = 0.0,
                                            regularization_type = self.regularization_type,
                                            output_type = self.output_type)
        '''

        # For Gear Mode (TBD)
        self.loss_function += tf.reduce_mean(tf.abs(self.y71)) * self.l1_norm

    # 入出力ベクトルの配置
    def make_feed_dict(self, prob, batch, is_train = True):
        feed_dict = {}
        feed_dict.setdefault(self.x, batch[0])
        #feed_dict.setdefault(self.y_, batch[1])
        feed_dict.setdefault(self.z_, batch[2])
        feed_dict.setdefault(self.learning_rate, self.learning_rate_value)
        feed_dict.setdefault(self.istraining, is_train)
        #feed_dict.setdefault(self.GearLevel, self.GearLevelValue)
        i = 0
        for keep_prob in self.keep_probs:
            if prob:
                feed_dict.setdefault(keep_prob['var'], 1.0)
            else:
                feed_dict.setdefault(keep_prob['var'], keep_prob['prob'])
            i += 1
        return feed_dict


    def get_auc(self, test, prob):
        fpr, tpr, thresholds = roc_curve(test, prob, pos_label = 1)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    def learning(self, data, save_at_log = False, validation_batch_num = 40):
        s = time.time()
        for i in range(self.epoch):
            batch = data.train.next_batch(self.batch)
            # 途中経過のチェック
            if i%self.log == 0 and i != 0:
                # Train
                feed_dict = self.make_feed_dict(prob = True, batch = batch, is_train = True)
                #train_accuracy_y = self.accuracy_y.eval(feed_dict=feed_dict)
                train_accuracy_z = self.accuracy_z.eval(feed_dict=feed_dict)
                losses = self.loss_function.eval(feed_dict=feed_dict)
                #train_prediction = self.prediction(data = batch[0], roi = False)
                #test = [batch[1][j][0] for j in range(len(batch[1]))]
                #prob = [train_prediction[0][j][0] for j in range(len(train_prediction[0]))]
                #train_auc = self.get_auc(test = test, prob = prob)
                # Test
                val_accuracy_y, val_accuracy_z, val_losses, test, prob = [], [], [], [], []
                for num in range(validation_batch_num):
                    validation_batch = data.test.next_batch(self.batch, augment = False)
                    feed_dict_val = self.make_feed_dict(prob = False, batch = validation_batch, is_train = True)
                    #val_accuracy_y.append(self.accuracy_y.eval(feed_dict=feed_dict_val) * float(self.batch))
                    val_accuracy_z.append(self.accuracy_z.eval(feed_dict=feed_dict_val) * float(self.batch))
                    val_losses.append(self.loss_function.eval(feed_dict=feed_dict_val) * float(self.batch))
                    #val_prediction = self.prediction(data = validation_batch[0], roi = False)
                    #test.extend([validation_batch[1][j][0] for j in range(len(validation_batch[1]))])
                    #prob.extend([val_prediction[0][j][0] for j in range(len(val_prediction[0]))])
                #val_accuracy_y = np.mean(val_accuracy_y) / float(self.batch)
                val_accuracy_z = np.mean(val_accuracy_z) / float(self.batch)
                val_losses = np.mean(val_losses) / float(self.batch)
                #val_auc = self.get_auc(test = test, prob = prob)

                # Output
                logger.debug("step %d ================================================================================="% i)
                #logger.debug("Train: (judgement, diagnosis, loss, auc) = (%g, %g, %g, %g)"%(train_accuracy_y,train_accuracy_z,losses,train_auc))
                #logger.debug("Validation: (judgement, diagnosis, loss, auc) = (%g, %g, %g, %g)"%(val_accuracy_y,val_accuracy_z,val_losses,val_auc))
                logger.debug("Train: (diagnosis, loss) = (%g, %g)"%(train_accuracy_z,losses))
                logger.debug("Validation: (diagnosis, loss) = (%g, %g)"%(val_accuracy_z, val_losses))

                if save_at_log:
                    self.save_checkpoint()
                e = time.time()
                elasped = e - s
                logger.debug("elasped time: %g" % elasped)
                s = e

            # 学習
            feed_dict = self.make_feed_dict(prob = False, batch = batch, is_train = True)
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
                   filenames = None, paths = None):
        # Make feed dict for prediction
        feed_dict = {self.x : data,
                     self.istraining: False}
        for keep_prob in self.keep_probs:
            feed_dict.setdefault(keep_prob['var'], 1.0)

        if self.output_type.find('hinge') >= 0:
            #result_y = self.sess.run(2.0 * self.y - 1.0, feed_dict = feed_dict)
            result_z = self.sess.run(2.0 * self.z - 1.0, feed_dict = feed_dict)
        else:
            #result_y = self.sess.run(tf.nn.softmax(self.y), feed_dict = feed_dict)
            result_z = self.sess.run(tf.sigmoid(self.z), feed_dict = feed_dict)
        result_y = [[1, 0] for i in range(len(paths))]
        if not roi:
            return result_y, result_z
        else:
            weights = self.get_output_weights(feed_dict = feed_dict)
            roi_base = self.get_roi_map_base(feed_dict = feed_dict)
            for i in range(len(paths)):
                self.make_roi(weights = weights[0],
                              roi_base = roi_base[0][i, :, :, :],
                              save_dir = save_dir,
                              filename = filenames[i],
                              label_def = label_def,
                              path = paths[i])

            return result_y, result_z

    def make_roi(self, weights, roi_base, save_dir, filename, label_def, path):
        img, bits = dicom_to_np(path)
        img = img / bits * 255
        img = img.astype(np.uint8)
        img = cv2.resize(img, (self.SIZE, self.SIZE), interpolation = cv2.INTER_AREA)
        img = np.stack((img, img, img), axis = -1)
        for x, finding in enumerate(label_def):
            images = np.zeros((roi_base.shape[0], roi_base.shape[1], 3))
            for channel in range(roi_base.shape[2]):
                c = roi_base[:, :, channel]
                image = np.stack((c, c, c), axis = -1)
                images += image * weights[channel][x]
            images = 255.0 * (images - np.min(images)) / (np.max(images) - np.min(images))
            images = cv2.applyColorMap(images.astype(np.uint8), cv2.COLORMAP_JET)
            images = cv2.resize(images, (self.SIZE, self.SIZE))
            roi_img = cv2.addWeighted(img, 0.7, images, 0.3, 1.0)
            cv2.imwrite(save_dir + '/' + str(filename[0]) + '_' + str(finding) + '.png', roi_img)
