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
from LinearMotor import Transfer as trans
from SimpleCells import *
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
        self.rmax = tf.placeholder(tf.float32)
        self.dmax = tf.placeholder(tf.float32)
        self.steps = 0
        self.val_losses = []
        self.current_loss = 0.0

    def construct(self):

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
        non_p_vars = list(set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) - self.p.model_weights_tensors)
        self.training(var_list = non_p_vars)
        logger.debug("05: TF Training operation done")
        # 精度の定義
        #self.accuracy_y = UT.correct_rate(self.y, self.y_)
        if self.output_type.find('hinge') >= 0:
            self.accuracy_z = tf.sqrt(tf.reduce_mean(tf.multiply(self.z - self.z_, self.z - self.z_)))
        else:
            self.accuracy_z = tf.sqrt(tf.reduce_mean(tf.multiply(tf.sigmoid(self.z) -self.z_, tf.sigmoid(self.z) -self.z_)))

        logger.debug("06: TF Accuracy measure definition done")
        # セッションの定義
        self.sess = tf.InteractiveSession()
        # チェックポイントの呼び出し
        self.saver = tf.train.Saver()
        self.restore()
        logger.debug("07: TF Model file definition done")
        self.p.load_weights()




    def io_def(self):
        self.CH = 3
        self.x = tf.placeholder("float", shape=[None, self.SIZE, self.SIZE, self.CH], name = "Input")
        #self.y_ = tf.placeholder("float", shape=[None, 2], name = "Label_Judgement")
        self.z_ = tf.placeholder("float", shape=[None, 14], name = "Label_Diagnosis")
        self.keep_probs = []

    def network(self):
        Initializer = 'He'
        Activation = 'Relu'
        Regularization = False
        Renormalization = False
        SE = True
        GrowthRate = 32
        prob = 1.0
        self.x_resnet = tf.image.resize_images(images = self.x,
                                               size = (224, 224),
                                               align_corners=False)
        self.p = trans.Transfer(self.x_resnet, 'resnet', pooling = None, vname = 'Transfer',
                                trainable = False)
        # activation_1 Tensor("Transfer/activation/Relu:0", shape=(1, 112, 112, 64), dtype=float32)
        #max_pooling2d_1 Tensor("Transfer/max_pooling2d/MaxPool:0", shape=(1, 55, 55, 64), dtype=float32)
        self.resnet_output = self.p.get_output_tensor()

        # Resnet Stem
        self.resnet_top = self.p['activation_1']
        self.y00 = Layers.pooling(x = self.resnet_top, ksize=[2, 2], strides=[2, 2],
                                  padding='SAME', algorithm = 'Max')
        self.resnet_stem = tf.image.resize_images(images = self.y00,
                                                  size = (self.SIZE / 4, self.SIZE / 4),
                                                  align_corners=False)

        # dense net
        ## Stem
        self.dense_stem = stem_cell(x = self.x,
                                    InputNode = [self.SIZE, self.SIZE, self.CH],
                                    Channels = 64,
                                    Initializer = Initializer,
                                    vname = 'Stem',
                                    regularization = Regularization,
                                    Training = self.istraining)

        ## concat
        self.stem_concat = Layers.concat([self.dense_stem, self.resnet_stem], concat_type = 'Channel')

        ## Dense
        self.densenet_output = densenet(x = self.stem_concat,
                                        Act = Activation,
                                        GrowthRate = GrowthRate,
                                        InputNode = [self.SIZE / 4, self.SIZE / 4, 128],
                                        Strides = [1, 1, 1, 1],
                                        Renormalization = Regularization,
                                        Regularization = Renormalization,
                                        rmax = None,
                                        dmax = None,
                                        SE = SE,
                                        Training = self.istraining,
                                        vname = 'DenseNet')
        self.resnet_output_resize = tf.image.resize_images(images = self.resnet_output,
                                                           size = (self.SIZE / 64, self.SIZE / 64),
                                                           align_corners=False)

        self.y41 = Layers.concat([self.resnet_output_resize, self.densenet_output], concat_type = 'Channel')

        # c = 2776
        c = (128 + GrowthRate * 16 + 2048) / 6
        self.y51 = inception_res_cell(x = self.y41,
                                      Act = Activation,
                                      InputNode = [self.SIZE / 64, self.SIZE / 64, 128 + GrowthRate * 16 + 2048],
                                      Channels0 = [c/4, c/4, c/4, c/4, c/4, c/4],
                                      Channels1 = [c, c, c, c, c, c],
                                      Strides0 = [1, 1, 1, 1],
                                      Strides1 = [1, 1, 1, 1],
                                      Initializer = Initializer,
                                      Regularization = Regularization,
                                      Renormalization = Renormalization,
                                      Rmax = self.rmax,
                                      Dmax = self.dmax,
                                      vname = 'Res51',
                                      SE = SE,
                                      Training = self.istraining)
        # Batch Normalization
        self.x_bn = Layers.batch_normalization(x = self.y51,
                                               shape = 128 + GrowthRate * 16 + 2048,
                                               vname = 'TOP_BN',
                                               dim = [0, 1, 2],
                                               Renormalization = Renormalization,
                                               Training = self.istraining,
                                               rmax = None,
                                               dmax = None)
        # Activation Function
        with tf.variable_scope('TOP_Act') as scope:
            self.x_act = AF.select_activation(Activation)(self.x_bn)


        self.y61 = Layers.pooling(x = self.y51,
                                  ksize=[self.SIZE / 64, self.SIZE / 64],
                                  strides=[self.SIZE / 64, self.SIZE / 64],
                                  padding='SAME',
                                  algorithm = 'Avg')


        # reshape
        self.y71 = Layers.reshape_tensor(x = self.y61, shape = [128 + GrowthRate * 16 + 2048])
        # fnn
        self.y72 = Outputs.output(x = self.y71,
                                  InputSize = 128 + GrowthRate * 16 + 2048,
                                  OutputSize = 14,
                                  Initializer = 'Xavier',
                                  BatchNormalization = False,
                                  Regularization = True,
                                  vname = 'Output_z')
        '''
        self.z0 = Layers.concat([self.y72, self.y71], concat_type = 'Vector')
        self.y73 = Outputs.output(x = self.z0,
                                  InputSize = 2048 + 14,
                                  OutputSize = 2,
                                  Initializer = 'Xavier',
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
                                             output_type = diag_output_type)
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
    def make_feed_dict(self, prob, batch, is_Train = True, is_update = False):
        if self.steps <= 5000:
            rmax, dmax = 1.0, 0.0
        else:
            rmax = min(1.0 + 2.0 * (40000.0 - float(self.steps)) / 40000.0, 3.0)
            dmax = min(5.0 * (25000.0 - float(self.steps)) / 25000.0, 5.0)

        if self.current_loss > np.mean(self.val_losses) - np.std(self.val_losses) and len(self.val_losses) > 10 and is_update:
            logger.debug("Before Learning Rate: %g" % self.learning_rate_value)
            self.learning_rate_value = max(0.00001, self.learning_rate_value * 0.9)
            logger.debug("After Learning Rate: %g" % self.learning_rate_value)
            self.val_losses = []
        feed_dict = {}
        feed_dict.setdefault(self.x, batch[0])
        #feed_dict.setdefault(self.y_, batch[1])
        feed_dict.setdefault(self.z_, batch[2])
        feed_dict.setdefault(self.learning_rate, self.learning_rate_value)
        feed_dict.setdefault(self.istraining, is_Train)
        feed_dict.setdefault(self.rmax, rmax)
        feed_dict.setdefault(self.dmax, dmax)
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

    def learning(self, data, save_at_log = False, validation_batch_num = 1):
        s = time.time()
        for i in range(self.epoch):
            batch = data.train.next_batch(self.batch, batch_ratio = 0.2)
            # 途中経過のチェック
            if i%self.log == 0 and i != 0:
                # Train
                self.p.change_phase(True)
                feed_dict = self.make_feed_dict(prob = True, batch = batch, is_Train = True)
                train_accuracy_z = self.accuracy_z.eval(feed_dict=feed_dict)
                losses = self.loss_function.eval(feed_dict=feed_dict)
                train_prediction = self.prediction(data = batch[0], roi = False)
                aucs_t = ''
                for d in range(len(train_prediction[1][0])):
                    test = [batch[2][j][d] for j in range(len(batch[2]))]
                    prob = [train_prediction[1][j][d] for j in range(len(train_prediction[1]))]
                    train_auc = self.get_auc(test = test, prob = prob)
                    aucs_t += "%03.2f / " % train_auc

                # Test
                self.p.change_phase(False)
                val_accuracy_y, val_accuracy_z, val_losses, test, prob = [], [], [], [], []
                validation_batch = data.test.next_batch(self.batch, augment = False)
                feed_dict_val = self.make_feed_dict(prob = False, batch = validation_batch, is_Train = False)
                val_accuracy_z = self.accuracy_z.eval(feed_dict=feed_dict_val)
                val_losses = self.loss_function.eval(feed_dict=feed_dict_val)
                val_prediction = self.prediction(data = validation_batch[0], roi = False)
                aucs_v = ''
                for d in range(len(train_prediction[1][0])):
                    test = [validation_batch[2][j][d] for j in range(len(validation_batch[2]))]
                    prob = [val_prediction[1][j][d] for j in range(len(val_prediction[1]))]
                    val_auc = self.get_auc(test = test, prob = prob)
                    aucs_v += "%03.2f / " % val_auc
                self.val_losses.append(val_losses)
                self.current_loss = val_losses

                # Output
                logger.debug("step %d ================================================================================="% i)
                #logger.debug("Train: (judgement, diagnosis, loss, auc) = (%g, %g, %g, %g)"%(train_accuracy_y,train_accuracy_z,losses,train_auc))
                #logger.debug("Validation: (judgement, diagnosis, loss, auc) = (%g, %g, %g, %g)"%(val_accuracy_y,val_accuracy_z,val_losses,val_auc))
                logger.debug("Train: (diagnosis, loss, aucs) = (%g, %g, %s)"%(train_accuracy_z,losses, aucs_t))
                logger.debug("Validation: (diagnosis, loss, aucs) = (%g, %g, %s)"%(val_accuracy_z, val_losses, aucs_v))

                if save_at_log:
                    self.save_checkpoint()
                e = time.time()
                elasped = e - s
                logger.debug("elasped time: %g" % elasped)
                s = e

            # 学習
            feed_dict = self.make_feed_dict(prob = False, batch = batch, is_Train = True, is_update = True)
            if self.DP and i != 0:
                self.dynamic_learning_rate(feed_dict)
            self.p.change_phase(True)
            self.train_op.run(feed_dict=feed_dict)
            self.steps += 1
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
        self.p.change_phase(False)
        # Make feed dict for prediction
        feed_dict = {self.x : data,
                     self.istraining : False}
        for keep_prob in self.keep_probs:
            feed_dict.setdefault(keep_prob['var'], 1.0)

        if self.output_type.find('hinge') >= 0:
            #result_y = self.sess.run(2.0 * self.y - 1.0, feed_dict = feed_dict)
            result_z = self.sess.run(2.0 * self.z - 1.0, feed_dict = feed_dict)
        else:
            #result_y = self.sess.run(tf.nn.softmax(self.y), feed_dict = feed_dict)
            result_z = self.sess.run(tf.sigmoid(self.z), feed_dict = feed_dict)
        result_y = [[1, 0] for i in range(len(result_z))]
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
