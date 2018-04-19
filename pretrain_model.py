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
        #non_p_vars = list(set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) - self.p.model_weights_tensors)
        #self.training(var_list = None)
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
        self.p.load_weights()




    def io_def(self):
        self.CH = 3
        self.x = tf.placeholder("float", shape=[None, self.SIZE, self.SIZE, self.CH], name = "Input")
        self.z_ = tf.placeholder("float", shape=[None, 15], name = "Label_Diagnosis")
        self.keep_probs = []

    def network(self):
        self.p = trans.Transfer(self.x, 'densenet201', pooling = None, vname = 'Transfer',
                                trainable = False)
        self.y51 = self.p.get_output_tensor()
        print(self.y51.shape)
        self.y61 = Layers.pooling(x = self.y51,
                                  ksize=[7, 7],
                                  strides=[7, 7],
                                  padding='SAME',
                                  algorithm = 'Avg')


        # reshape
        self.y71 = Layers.reshape_tensor(x = self.y61, shape = [1 * 1 * 1920])
        # fnn
        self.y72 = Outputs.output(x = self.y71,
                                  InputSize = 1920,
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
                                             regularization = self.regularization,
                                             regularization_type = self.regularization_type,
                                             output_type = diag_output_type)

        # For Gear Mode (TBD)
        self.loss_function += tf.reduce_mean(tf.abs(self.y71)) * self.l1_norm
        vs.variable_summary(self.loss_function, 'Loss', is_scalar = True)

    # 学習
    def training(self, var_list = None, gradient_cliiping = True, clipping_norm = 1.0):
        self.train_op, self.optimizer = TO.select_algo(loss_function = self.loss_function,
                                                       algo = self.optimizer_type,
                                                       learning_rate = self.learning_rate,
                                                       b1 = self.beta1, b2 = self.beta2,
                                                        var_list = var_list,
                                                        gradient_cliiping = gradient_cliiping,
                                                        clipping_norm = clipping_norm)
        self.grad_op = self.optimizer.compute_gradients(self.loss_function)

    # 入出力ベクトルの配置
    def make_feed_dict(self, prob, batch):
        feed_dict = {}
        feed_dict.setdefault(self.x, batch[0])
        feed_dict.setdefault(self.z_, batch[2])
        feed_dict.setdefault(self.learning_rate, self.learning_rate_value)
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
            batch = data.train.next_batch(self.batch)
            # 途中経過のチェック
            if i%self.log == 0 and i != 0:
                # Train
                self.p.change_phase(True)
                feed_dict = self.make_feed_dict(prob = True, batch = batch)
                res = self.sess.run([self.accuracy_z, self.loss_function], feed_dict = feed_dict)
                train_accuracy_z = res[0]
                losses = res[1]
                #l1_losses = res[2]
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
                validation_batch = data.test.next_batch(self.batch, augment = False, batch_ratio = batch_ratio[i % len(batch_ratio)])
                feed_dict_val = self.make_feed_dict(prob = True, batch = validation_batch)
                res_val = self.sess.run([self.accuracy_z, self.loss_function], feed_dict = feed_dict_val)
                val_accuracy_z = res_val[0]
                val_losses = res_val[1]
                #val_l1 = res_val[2]
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
            feed_dict = self.make_feed_dict(prob = False, batch = batch)
            if self.DP and i != 0:
                self.dynamic_learning_rate(feed_dict)
            self.p.change_phase(True)
            _, summary = self.sess.run([self.train_op, self.summary], feed_dict=feed_dict)
            vs.add_log(writer = self.train_writer, summary = summary, step = i)
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
        self.p.change_phase(False)
        # Make feed dict for prediction
        feed_dict = {self.x : data}
        for keep_prob in self.keep_probs:
            feed_dict.setdefault(keep_prob['var'], 1.0)

        if self.output_type.find('hinge') >= 0:
            result_z = self.sess.run(2.0 * self.z - 1.0, feed_dict = feed_dict)
        else:
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
