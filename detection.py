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
from LinearMotor import Visualizer as vs
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
        self.l1_norm = tf.placeholder(tf.float32)
        self.regularization = tf.placeholder(tf.float32)
        self.rmax = tf.placeholder(tf.float32, shape=())
        self.dmax = tf.placeholder(tf.float32, shape=())
        #self.rmax = tf.placeholder_with_default(1.0, shape=())
        #self.dmax = tf.placeholder_with_default(0.0, shape=())
        self.steps = 0
        self.val_losses = []
        self.current_loss = 0.0
        self.l1_norm_value = 0.0
        self.regularization_value = 0.0
        self.eval_l1_loss = 0.0


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
        self.training(var_list = None)
        logger.debug("05: TF Training operation done")
        # 精度の定義
        if self.output_type.find('hinge') >= 0:
            self.accuracy_z = tf.sqrt(tf.reduce_mean(tf.multiply(self.z - self.z_, self.z - self.z_)))
        else:
            self.accuracy_z = tf.sqrt(tf.reduce_mean(tf.multiply(tf.sigmoid(self.z) -self.z_, tf.sigmoid(self.z) -self.z_)))
        vs.variable_summary(self.loss_function, 'Accuracy', is_scalar = True)

        logger.debug("06: TF Accuracy measure definition done")
        # セッションの定義
        self.sess = tf.InteractiveSession()
        # tensor board
        now = datetime.now()
        now = now.strftime("%Y-%m-%d")
        self.summary, self.train_writer, self.test_writer = vs.file_writer(sess = self.sess, file_name = './Result/' + now)
        # チェックポイントの呼び出し
        self.saver = tf.train.Saver()
        self.restore()
        logger.debug("07: TF Model file definition done")




    def io_def(self):
        self.CH = 3
        self.x = tf.placeholder("float", shape=[None, self.SIZE, self.SIZE, self.CH], name = "Input")
        self.z_ = tf.placeholder("float", shape=[None, 15], name = "Label_Diagnosis")
        self.keep_probs = []

    def network(self):
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
        ## Stem
        # Batch Normalization
        self.stem_bn = Layers.batch_normalization(x = self.x,
                                                  shape = self.CH,
                                                  vname = 'STEM_TOP_BN01',
                                                  dim = [0, 1, 2],
                                                  Renormalization = Renormalization,
                                                  Training = self.istraining,
                                                  rmax = self.rmax,
                                                  dmax = self.dmax)
        self.dense_stem = stem_cell(x = self.stem_bn,
                                    InputNode = [self.SIZE, self.SIZE, self.CH],
                                    Channels = StemChannels,
                                    Initializer = Initializer,
                                    vname = 'Stem',
                                    regularization = Regularization,
                                    Training = self.istraining)

        ## Dense
        self.densenet_output = densenet(x = self.dense_stem,
                                        root = self.stem_bn,
                                        Act = Activation,
                                        GrowthRate = GrowthRate,
                                        InputNode = [self.SIZE / 4, self.SIZE / 4, StemChannels],
                                        Strides = [1, 1, 1, 1],
                                        Renormalization = Renormalization,
                                        Regularization = Regularization,
                                        rmax = self.rmax,
                                        dmax = self.dmax,
                                        SE = SE,
                                        Training = self.istraining,
                                        GroupNorm = GroupNorm,
                                        GroupNum = GroupNum,
                                        vname = 'DenseNet')

        self.y50 = self.densenet_output
        self.y51 = SE_module(x = self.y50,
                             InputNode = [self.SIZE / 64, self.SIZE / 64, StemChannels + 12 +GrowthRate * 98],
                             Act = Activation,
                             Rate = 0.5,
                             vname = 'TOP_SE')

        self.y61 = Layers.pooling(x = self.y51,
                                  ksize=[self.SIZE / 64, self.SIZE / 64],
                                  strides=[self.SIZE / 64, self.SIZE / 64],
                                  padding='SAME',
                                  algorithm = 'Avg')

        # reshape
        self.y71 = Layers.reshape_tensor(x = self.y61, shape = [StemChannels + 12 +GrowthRate * 98])
        self.y71_d = Layers.dropout(x = self.y71,
                                    keep_probs = self.keep_probs,
                                    training_prob = prob,
                                    vname = 'Dropout')
        # fnn
        self.y72 = Outputs.output(x = self.y71_d,
                                  InputSize = StemChannels + 12 + GrowthRate * 98,
                                  OutputSize = 15,
                                  Initializer = 'Xavier',
                                  BatchNormalization = False,
                                  Regularization = True,
                                  vname = 'Output_z')
        self.z = self.y72


    def loss(self):
        diag_output_type = self.output_type if self.output_type.find('hinge') >= 0 else 'classified-sigmoid'
        self.loss_function = Loss.loss_func(y = self.z,
                                             y_ = self.z_,
                                             regularization = 0.0,
                                             regularization_type = self.regularization_type,
                                             output_type = diag_output_type)
        vs.variable_summary(self.loss_function, 'Loss', is_scalar = True)
        #self.l2_loss = 0
        # For Gear Mode (TBD)
        #self.l1_loss = tf.reduce_mean(tf.abs(self.y71)) * self.l1_norm
        #self.l2_loss= tf.sqrt(tf.reduce_mean(tf.multiply(tf.sigmoid(self.z) -self.z_, tf.sigmoid(self.z) -self.z_))) * 0.5
        #self.loss_function += self.l2_loss

    def training(self, var_list = None, gradient_cliiping = True, clipping_norm = 0.1):
        self.train_op, self.optimizer = TO.select_algo(loss_function = self.loss_function,
                                                        algo = self.optimizer_type,
                                                        learning_rate = self.learning_rate,
                                                        b1 = self.beta1, b2 = self.beta2,
                                                        var_list = var_list,
                                                        gradient_cliiping = gradient_cliiping,
                                                        clipping_norm = clipping_norm)
        self.grad_op = self.optimizer.compute_gradients(self.loss_function)


    # 入出力ベクトルの配置
    def make_feed_dict(self, prob, batch, is_Train = True, is_update = False):
        if self.steps <= 5000:
            rmax, dmax = 1.0, 0.0
        else:
            rmax = min(1.0 + 2.0 * float(self.steps - 5000.0) / 35000.0, 3.0)
            dmax = min(5.0 * float(self.steps -5000.0) / 20000.0, 5.0)
        if self.steps % 3000 == 0 and self.steps != 0 and is_update:
            logger.debug("Before Learning Rate: %g" % self.learning_rate_value)
            self.learning_rate_value = max(0.000001, self.learning_rate_value * 0.9)
            logger.debug("After Learning Rate: %g" % self.learning_rate_value)

        feed_dict = {}
        feed_dict.setdefault(self.x, batch[0])
        feed_dict.setdefault(self.z_, batch[2])
        feed_dict.setdefault(self.learning_rate, self.learning_rate_value)
        feed_dict.setdefault(self.istraining, is_Train)
        feed_dict.setdefault(self.rmax, rmax)
        feed_dict.setdefault(self.dmax, dmax)
        feed_dict.setdefault(self.regularization, self.regularization_value)
        feed_dict.setdefault(self.l1_norm, self.l1_norm_value)
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

    def learning(self, data, save_at_log = False, validation_batch_num = 1, batch_ratio = [0.2, 0.3, 0.4, 0.5, 0.6]):
        s = time.time()
        for i in range(self.epoch):
            batch = data.train.next_batch(self.batch, batch_ratio = batch_ratio[i % len(batch_ratio)])
            # 途中経過のチェック
            if i%self.log == 0 and i != 0:
                # Train
                feed_dict = self.make_feed_dict(prob = True, batch = batch, is_Train = True)
                res = self.sess.run([self.accuracy_z, self.loss_function], feed_dict = feed_dict)
                train_accuracy_z = res[0]
                losses = res[1]
                train_prediction = self.prediction(data = batch[0], roi = False)
                aucs_t = ''
                for d in range(len(train_prediction[1][0])):
                    test = [batch[2][j][d] for j in range(len(batch[2]))]
                    prob = [train_prediction[1][j][d] for j in range(len(train_prediction[1]))]
                    train_auc = self.get_auc(test = test, prob = prob)
                    aucs_t += "%03.2f / " % train_auc

                # Test
                val_accuracy_y, val_accuracy_z, val_losses, test, prob = [], [], [], [], []
                validation_batch = data.test.next_batch(self.batch, augment = False, batch_ratio = batch_ratio[i % len(batch_ratio)])
                feed_dict_val = self.make_feed_dict(prob = True, batch = validation_batch, is_Train = False)
                res_val = self.sess.run([self.accuracy_z, self.loss_function], feed_dict = feed_dict_val)
                val_accuracy_z = res_val[0]
                val_losses = res_val[1]
                val_prediction = self.prediction(data = validation_batch[0], roi = False)
                aucs_v = ''
                for d in range(len(train_prediction[1][0])):
                    test = [validation_batch[2][j][d] for j in range(len(validation_batch[2]))]
                    prob = [val_prediction[1][j][d] for j in range(len(val_prediction[1]))]
                    val_auc = self.get_auc(test = test, prob = prob)
                    aucs_v += "%03.2f / " % val_auc
                self.val_losses.append(val_losses)
                self.current_loss = val_losses
                #self.eval_l1_loss = min(val_l1, l1_losses)

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
            _, summary = self.sess.run([self.train_op, self.summary], feed_dict=feed_dict)
            vs.add_log(writer = self.train_writer, summary = summary, step = i)
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
                   filenames = None, findings = None, roi_force = False):
        # Make feed dict for prediction
        if self.steps <= 5000:
            rmax, dmax = 1.0, 0.0
        else:
            rmax = min(1.0 + 2.0 * float(self.steps - 5000.0) / 35000.0, 3.0)
            dmax = min(5.0 * float(self.steps -5000.0) / 20000.0, 5.0)
        feed_dict = {self.x : data,
                     self.istraining : False,
                     self.rmax : rmax,
                     self.dmax : dmax}

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
            for i in range(len(filenames)):
                self.make_roi(weights = weights[0],
                              roi_base = roi_base[0][i, :, :, :],
                              save_dir = save_dir,
                              filename = filenames[i],
                              label_def = label_def,
                              findings = findings[i],
                              roi_force = roi_force)

            return result_y, result_z

    def make_roi(self, weights, roi_base, save_dir, filename, label_def, findings,
                 roi_force):
        if filename.find('.png') >= 0:
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        else:
            img, bits = dicom_to_np(filename)
            img = img / bits * 255
            img = img.astype(np.uint8)
        img = cv2.resize(img, (self.SIZE, self.SIZE), interpolation = cv2.INTER_AREA)
        img = np.stack((img, img, img), axis = -1)
        for x, finding in enumerate(label_def):
            images = np.zeros((roi_base.shape[0], roi_base.shape[1], 3))
            for channel in range(roi_base.shape[2]):
                c = roi_base[:, :, channel]
                c0 = np.zeros((roi_base.shape[0], roi_base.shape[1]))
                image = np.stack((c, c, c), axis = -1)
                images += image * weights[channel][x]
            images = np.maximum(images - np.mean(images), 0)
            images = 255.0 * (images - np.min(images)) / (np.max(images) - np.min(images))
            #images = 255.0 * (images) / np.max(images)
            images = cv2.applyColorMap(images.astype(np.uint8), cv2.COLORMAP_JET)
            images = cv2.resize(images.astype(np.uint8), (self.SIZE, self.SIZE))
            roi_img = cv2.addWeighted(img, 0.8, images, 0.2, 0)
            basename = os.path.basename(filename)
            ftitle, _ = os.path.splitext(basename)
            if roi_force:
                if filename.find('.png') >= 0:
                    cv2.imwrite(save_dir + '/' + str(ftitle) + '_' + str(finding) + '_' + findings + '.png', roi_img)
                else:
                    cv2.imwrite(save_dir + '/' + str(ftitle) + '_' + str(finding) + '.png', roi_img)
            else:
                if findings.find(finding) >= 0 or filename.find('.dcm') >= 0:
                    if filename.find('.png') >= 0:
                        cv2.imwrite(save_dir + '/' + str(ftitle) + '_' + str(finding) + '_' + findings + '.png', roi_img)
                    else:
                        if finding in ['Nodule', 'Mass', 'Pneumonia']:
                            cv2.imwrite(save_dir + '/' + str(ftitle) + '_' + str(finding) + '.png', roi_img)
