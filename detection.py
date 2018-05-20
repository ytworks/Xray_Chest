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
from NetworkModules import *
from logging import getLogger, StreamHandler
logger = getLogger(__name__)
sh = StreamHandler()
logger.addHandler(sh)
logger.setLevel(10)


class Detecter(Core2.Core):
    def __init__(self,
                 output_type,
                 epoch=100, batch=32, log=10,
                 optimizer_type='Adam',
                 learning_rate=0.0001,
                 dynamic_learning_rate=0.000001,
                 beta1=0.9, beta2=0.999,
                 dumping_period=9000,
                 dumping_rate=0.9,
                 regularization=0.0,
                 regularization_type='L2',
                 checkpoint='./Storages/Core.ckpt',
                 init=True,
                 size=256,
                 l1_norm=0.1,
                 step=0,
                 network_mode='scratch'):
        super(Detecter, self).__init__(output_type=output_type,
                                       epoch=epoch,
                                       batch=batch,
                                       log=log,
                                       optimizer_type=optimizer_type,
                                       learning_rate=learning_rate,
                                       dynamic_learning_rate=dynamic_learning_rate,
                                       beta1=beta1,
                                       beta2=beta2,
                                       regularization=regularization,
                                       regularization_type=regularization_type,
                                       checkpoint=checkpoint,
                                       init=init
                                       )
        self.SIZE = size
        self.l1_norm = tf.placeholder(tf.float32)
        self.regularization = tf.placeholder(tf.float32)
        self.rmax = tf.placeholder(tf.float32, shape=())
        self.dmax = tf.placeholder(tf.float32, shape=())
        self.steps = step
        self.l1_norm_value = 0.0
        self.regularization_value = 0.0
        self.dumping_rate = dumping_rate
        self.dumping_period = dumping_period
        self.network_mode = network_mode
        for i in range(self.steps):
            if i != 0 and i % self.dumping_period == 0:
                self.learning_rate_value = max(
                    0.000001, self.learning_rate_value * self.dumping_rate)

        logger.info("start step %g, learning_rate %g" %
                    (self.steps, self.learning_rate_value))

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
        self.training(var_list=None)
        logger.debug("05: TF Training operation done")
        # 精度の定義
        if self.output_type.find('hinge') >= 0:
            self.accuracy_z = tf.sqrt(tf.reduce_mean(
                tf.multiply(self.z - self.z_, self.z - self.z_)))
        else:
            self.accuracy_z = tf.sqrt(tf.reduce_mean(tf.multiply(
                tf.sigmoid(self.z) - self.z_, tf.sigmoid(self.z) - self.z_)))
        vs.variable_summary(self.accuracy_z, 'Accuracy', is_scalar=True)

        logger.debug("06: TF Accuracy measure definition done")
        # セッションの定義
        self.sess = tf.InteractiveSession()
        # tensor board
        now = datetime.now()
        now = now.strftime("%Y-%m-%d")
        self.summary, self.train_writer, self.test_writer = vs.file_writer(
            sess=self.sess, file_name='./Result/' + now)
        # チェックポイントの呼び出し
        self.saver = tf.train.Saver()
        self.restore()
        logger.debug("07: TF Model file definition done")

    def io_def(self):
        self.CH = 3
        self.x = tf.placeholder(
            "float", shape=[None, self.SIZE, self.SIZE, self.CH], name="Input")
        self.z_ = tf.placeholder(
            "float", shape=[None, 15], name="Label_Diagnosis")
        self.keep_probs = []

    def network(self):
        if self.network_mode == 'scratch':
            self.z, self.logit, self.y51 = scratch_model(x=self.x,
                                                         SIZE=self.SIZE,
                                                         CH=self.CH,
                                                         istraining=self.istraining,
                                                         rmax=self.rmax,
                                                         dmax=self.dmax,
                                                         keep_probs=self.keep_probs)
        elif self.network_mode == 'pretrain':
            self.z, self.logit, self.y51, self.p = pretrain_model(x=self.x)
        else:
            self.z, self.logit, self.y51 = scratch_model(x=self.x,
                                                         SIZE=self.SIZE,
                                                         CH=self.CH,
                                                         istraining=self.istraining,
                                                         rmax=self.rmax,
                                                         dmax=self.dmax,
                                                         keep_probs=self.keep_probs)

    def loss(self):
        diag_output_type = self.output_type if self.output_type.find(
            'hinge') >= 0 else 'classified-sigmoid'
        self.loss_function = Loss.loss_func(y=self.z,
                                            y_=self.z_,
                                            regularization=0.0,
                                            regularization_type=self.regularization_type,
                                            output_type=diag_output_type)
        vs.variable_summary(self.loss_function, 'Loss', is_scalar=True)

    def training(self, var_list=None, gradient_cliiping=True, clipping_norm=0.1):
        self.train_op, self.optimizer = TO.select_algo(loss_function=self.loss_function,
                                                       algo=self.optimizer_type,
                                                       learning_rate=self.learning_rate,
                                                       b1=self.beta1, b2=self.beta2,
                                                       var_list=var_list,
                                                       gradient_cliiping=gradient_cliiping,
                                                       clipping_norm=clipping_norm)
        self.grad_op = self.optimizer.compute_gradients(self.loss_function)

    # 入出力ベクトルの配置

    def make_feed_dict(self, prob, data, label=None, is_Train=True, is_update=False, is_label=False):
        if self.steps <= 5000:
            rmax, dmax = 1.0, 0.0
        else:
            rmax = min(1.0 + 2.0 * float(self.steps - 5000.0) / 35000.0, 3.0)
            dmax = min(5.0 * float(self.steps - 5000.0) / 20000.0, 5.0)
        if self.steps % self.dumping_period == 0 and self.steps != 0 and is_update:
            logger.debug("Before Learning Rate: %g" % self.learning_rate_value)
            self.learning_rate_value = max(
                0.000001, self.learning_rate_value * self.dumping_rate)
            logger.debug("After Learning Rate: %g" % self.learning_rate_value)

        feed_dict = {}
        feed_dict.setdefault(self.x, data)
        if is_label:
            feed_dict.setdefault(self.z_, label)
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
        fpr, tpr, thresholds = roc_curve(test, prob, pos_label=1)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    def get_auc_list(self, feed_dict, batch):
        res = self.sess.run(
            [self.accuracy_z, self.loss_function], feed_dict=feed_dict)
        accuracy_z = res[0]
        losses = res[1]
        prediction = self.prediction(data=batch[0], roi=False)
        aucs_list = ''
        for d in range(len(prediction[1][0])):
            test = [batch[2][j][d] for j in range(len(batch[2]))]
            prob = [prediction[1][j][d]
                    for j in range(len(prediction[1]))]
            auc_value = self.get_auc(test=test, prob=prob)
            aucs_list += "%03.2f / " % auc_value
        return accuracy_z, losses, aucs_list

    def learning(self, data, save_at_log=False, validation_batch_num=1, batch_ratio=[0.2, 0.3, 0.4]):
        s = time.time()
        for i in range(self.epoch):
            batch = data.train.next_batch(
                self.batch, batch_ratio=batch_ratio[i % len(batch_ratio)])
            # 途中経過のチェック
            if i % self.log == 0 and i != 0:
                if self.network_mode == 'pretrain':
                    self.p.change_phase(False)
                # Train
                feed_dict = self.make_feed_dict(
                    prob=True, data=batch[0], label=batch[2], is_Train=False, is_label=True)
                train_accuracy_z, losses, aucs_t = self.get_auc_list(
                    feed_dict, batch)
                # Validation sample
                val_accuracy_y, val_accuracy_z, val_losses, test, prob = [], [], [], [], []
                validation_batch = data.val.next_batch(
                    self.batch, augment=False, batch_ratio=batch_ratio[i % len(batch_ratio)])
                feed_dict_val = self.make_feed_dict(
                    prob=True, data=validation_batch[0], label=validation_batch[2], is_Train=False, is_label=True)
                val_accuracy_z, val_losses, aucs_v = self.get_auc_list(
                    feed_dict_val, validation_batch)
                # Output
                logger.debug(
                    "step %d =================================================================================" % i)
                logger.debug("Train: (diagnosis, loss, aucs) = (%g, %g, %s)" % (
                    train_accuracy_z, losses, aucs_t))
                logger.debug("Validation: (diagnosis, loss, aucs) = (%g, %g, %s)" % (
                    val_accuracy_z, val_losses, aucs_v))
                if save_at_log:
                    self.save_checkpoint()
                e = time.time()
                elasped = e - s
                logger.debug("elasped time: %g" % elasped)
                s = e
            # 学習
            if self.network_mode == 'pretrain':
                self.p.change_phase(True)
            feed_dict = self.make_feed_dict(
                prob=False, data=batch[0], label=batch[2], is_Train=True, is_update=True, is_label=True)
            if self.DP and i != 0:
                self.dynamic_learning_rate(feed_dict)
            _, summary = self.sess.run(
                [self.train_op, self.summary], feed_dict=feed_dict)
            vs.add_log(writer=self.train_writer, summary=summary, step=i)
            validation_batch = data.val.next_batch(
                self.batch, augment=False, batch_ratio=batch_ratio[i % len(batch_ratio)])
            feed_dict_val = self.make_feed_dict(
                prob=True, data=validation_batch[0], label=validation_batch[2], is_Train=False, is_label=True)
            summary = self.sess.run(self.summary, feed_dict=feed_dict_val
                                    )
            vs.add_log(writer=self.test_writer, summary=summary, step=i)
            self.steps += 1
        self.save_checkpoint()

    def get_output_weights(self, feed_dict):
        tvar = self.get_trainable_var()
        output_vars = []
        for var in tvar:
            if var.name.find('Output_z') >= 0:
                output_vars.append(var)
        weights = self.sess.run(output_vars, feed_dict=feed_dict)
        return weights

    def get_roi_map_base(self, feed_dict):
        return self.sess.run([self.y51], feed_dict=feed_dict)

    # 予測器

    def prediction(self, data, roi=False, label_def=None, save_dir=None,
                   filenames=None, suffixs=None, roi_force=False):
        # Make feed dict for prediction
        if self.network_mode == 'pretrain':
            self.p.change_phase(False)
        feed_dict = self.make_feed_dict(
            prob=True, data=data, is_Train=False, is_label=False)
        # Get logits
        if self.output_type.find('hinge') >= 0:
            result_z = self.sess.run(2.0 * self.z - 1.0, feed_dict=feed_dict)
        else:
            result_z = self.sess.run(self.logit, feed_dict=feed_dict)
        result_y = [[1, 0] for i in range(len(result_z))]
        # Make ROI maps
        if not roi:
            return result_y, result_z, None
        else:
            weights = self.get_output_weights(feed_dict=feed_dict)
            roi_base = self.get_roi_map_base(feed_dict=feed_dict)
            result_roi = []
            for i in range(len(filenames)):
                roi_map = self.make_roi(weights=weights[0],
                              roi_base=roi_base[0][i, :, :, :],
                              save_dir=save_dir,
                              filename=filenames[i],
                              label_def=label_def,
                              suffix=suffixs[i],
                              roi_force=roi_force)
                result_roi.append(roi_map)

            return result_y, result_z, np.array(result_roi)

    def make_roi(self, weights, roi_base, save_dir, filename, label_def, suffix,
                 roi_force):
        # Read files
        if filename.find('.png') >= 0:
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        else:
            img, bits = dicom_to_np(filename)
            img = img / bits * 255
            img = img.astype(np.uint8)
        # Preprocessing
        img = cv2.resize(img, (self.SIZE, self.SIZE),
                         interpolation=cv2.INTER_AREA)
        img = np.stack((img, img, img), axis=-1)
        roi_maps = []
        for x, finding in enumerate(label_def):
            # sum channels
            images = np.zeros((roi_base.shape[0], roi_base.shape[1], 3))
            for channel in range(roi_base.shape[2]):
                c = roi_base[:, :, channel]
                c0 = np.zeros((roi_base.shape[0], roi_base.shape[1]))
                image = np.stack((c, c, c), axis=-1)
                images += image * weights[channel][x]
            # process image
            images = np.maximum(images - np.mean(images), 0)
            images = 255.0 * (images - np.min(images)) / \
                (np.max(images) - np.min(images))
            roi_map = cv2.resize(images.astype(np.uint8),
                                 (self.SIZE, self.SIZE))
            roi_maps.append(roi_map)
            # overlay original image
            images = cv2.applyColorMap(
                images.astype(np.uint8), cv2.COLORMAP_JET)
            images = cv2.resize(images.astype(np.uint8),
                                (self.SIZE, self.SIZE))
            roi_img = cv2.addWeighted(img, 0.8, images, 0.2, 0)
            # save image
            basename = os.path.basename(filename)
            ftitle, _ = os.path.splitext(basename)
            if roi_force:
                if filename.find('.png') >= 0:
                    cv2.imwrite(save_dir + '/' + str(ftitle) + '_' +
                                str(finding) + '_' + suffix + '.png', roi_img)
                else:
                    cv2.imwrite(save_dir + '/' + str(ftitle) +
                                '_' + str(finding) + '.png', roi_img)
            else:
                if findings.find(finding) >= 0 or filename.find('.dcm') >= 0:
                    if filename.find('.png') >= 0:
                        cv2.imwrite(save_dir + '/' + str(ftitle) + '_' +
                                    str(finding) + '_' + suffix + '.png', roi_img)
                    else:
                        if finding in ['Nodule', 'Mass', 'Pneumonia']:
                            cv2.imwrite(save_dir + '/' + str(ftitle) +
                                        '_' + str(finding) + '.png', roi_img)
        return np.array(roi_maps)
