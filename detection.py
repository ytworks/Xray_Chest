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


class Detector(Core2.Core):
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
                 network_mode='scratch',
                 transfer_save_mode=True,
                 transfer_checkpoint='./Strages/Core.ckpt',
                 tflog=10,
                 gpu_num=0,
                 distributed_batch=32,
                 config=None):
        super(Detector, self).__init__(output_type=output_type,
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
        self.CH = config.getint('DLParams', 'channel')
        self.l1_norm = tf.placeholder(tf.float32)
        self.regularization = tf.placeholder(tf.float32)
        self.rmax = tf.placeholder(tf.float32, shape=())
        self.dmax = tf.placeholder(tf.float32, shape=())
        self.wd = tf.placeholder(tf.float32, shape=())
        self.steps = step
        self.l1_norm_value = 0.0
        self.regularization_value = regularization
        self.dumping_rate = dumping_rate
        self.dumping_period = dumping_period
        self.network_mode = network_mode
        self.transfer_save_mode = transfer_save_mode
        self.transfer_checkpoint = transfer_checkpoint
        self.tflog = tflog
        self.gpu_num = gpu_num
        self.distributed_batch = distributed_batch
        self.prev_val = 10000.0
        self.gradient_init = 0
        self.config = config
        self.wd_value = self.config.getfloat('DLParams', 'weight_decay')
        self.t_cur = 0
        self.t_i = self.config.getfloat('DLParams', 't_i')
        print(self.optimizer_type)
        for i in range(self.steps):
            if i != 0 and i % self.dumping_period == 0:
                self.learning_rate_value = max(
                    1.0e-8, self.learning_rate_value * self.dumping_rate)
                self.wd_value = max(
                    1.0e-16, self.wd_value * self.dumping_rate)

        logger.info("start step %g, learning_rate %g" %
                    (self.steps, self.learning_rate_value))

    def construct(self):
        logger.debug("01: TF session Start")
        # 入出力の定義
        self.io_def()
        logger.debug("02: TF I/O definition done")
        # 学習
        if self.gpu_num < 2:
            self.training(var_list=None)
        else:
            self.multi_training(var_list=None)
        logger.debug("03: TF Training operation done")
        # セッションの定義
        with tf.device('/cpu:0'):
            config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=tf.GPUOptions(
                                        per_process_gpu_memory_fraction=0.8,  # 最大値の80%まで
                                        allow_growth=True  # True->必要になったら確保, False->全部
                                    ))
            self.sess = tf.InteractiveSession(config=config)
            # tensor board
            now = datetime.now()
            now = now.strftime("%Y-%m-%d")
            self.summary, self.train_writer, self.val_writer, self.test_writer = vs.file_writer(
                sess=self.sess, file_name=self.config.get('OutputParams', 'logfile') + '/' + now)
            # チェックポイントの呼び出し
            self.saver = tf.train.Saver(
                list(set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))))
            self.restore()
            logger.debug("05: TF Model file definition done")

    def save_transfer_checkpoint(self):
        UT.save_checkpoint(saver=self.transfer_saver,
                           checkpoint=self.transfer_checkpoint,
                           sess=self.sess)

    def save_checkpoint(self):
        UT.save_checkpoint(saver=self.saver,
                           checkpoint=self.checkpoint,
                           sess=self.sess)

    def validation_save(self, vname):
        ckpt = self.checkpoint.replace(".ckpt", vname + ".ckpt")
        t_ckpt = self.transfer_checkpoint.replace(".ckpt", vname + ".ckpt")
        UT.save_checkpoint(saver=self.saver,
                           checkpoint=ckpt,
                           sess=self.sess)

    def io_def(self):
        with tf.device('/cpu:0'):
            self.x = tf.placeholder(
                "float", shape=[None, self.SIZE, self.SIZE, self.CH], name="Input")
            self.z_ = tf.placeholder(
                "float", shape=[None, 15], name="Label_Diagnosis")
            self.keep_probs = []

    def loss(self, z, z_):
        diag_output_type = self.output_type
        loss_ce = Loss.loss_func(y=z,
                                 y_=z_,
                                 regularization=self.regularization,
                                 regularization_type=self.regularization_type,
                                 output_type=diag_output_type,
                                 alpha=self.config.getfloat(
                                     'DLParams', 'focal_alpha'),
                                 gamma=self.config.getfloat(
                                     'DLParams', 'focal_gamma')
                                 )
        with tf.device('/cpu:0'):
            vs.variable_summary(loss_ce, 'Loss', is_scalar=True)
        return loss_ce

    def training(self, var_list=None, gradient_cliiping=True, clipping_norm=0.01):
        vs.variable_summary(self.learning_rate, 'LearningRate')
        self.optimizer = TO.get_opt(algo=self.optimizer_type,
                                    learning_rate=self.learning_rate,
                                    b1=np.float32(self.beta1),
                                    b2=np.float32(self.beta2),
                                    nesterov=self.config.getboolean(
                                        'DLParams', 'nesterov'),
                                    weight_decay=self.wd)
        self.z, self.logit, self.y51 = light_model(x=self.x,
                                                   is_train=self.istraining,
                                                   rmax=self.rmax,
                                                   dmax=self.dmax,
                                                   ini=self.config,
                                                   reuse=False)
        self.loss_function = self.loss(z=self.z, z_=z_)
        grads = TO.get_grads(optimizer=self.optimizer,
                             loss_function=self.loss_function,
                             var_list=var_list,
                             gradient_clipping=gradient_cliiping,
                             clipping_norm=clipping_norm,
                             clipping_type='norm')
        self.train_op = TO.get_train_op(optimizer=self.optimizer,
                                        grad_var_pairs=grads,
                                        ema=False,
                                        ema_decay=0.9999)

    def multi_training(self, var_list=None, gradient_cliiping=True, clipping_norm=0.01):
        with tf.device('/cpu:0'):
            vs.variable_summary(self.learning_rate, 'LearningRate')
            self.optimizer = TO.get_opt(algo=self.optimizer_type,
                                        learning_rate=self.learning_rate,
                                        b1=np.float32(self.beta1),
                                        b2=np.float32(self.beta2),
                                        nesterov=self.config.getboolean(
                                            'DLParams', 'nesterov'),
                                        weight_decay=self.wd)
            logger.debug("03-01: Optimizer definition")
            batch_size = tf.shape(self.x)[0]
            effective_gpu_num = batch_size // self.distributed_batch
            m = [-1].extend([self.distributed_batch] * effective_gpu_num)
            xs = tf.split(self.x, m)
            z_s = tf.split(self.z_, m)
            logger.debug("03-03: Data split")
            tower_grads = []
            self.losses, self.logits, self.y51s = [], [], []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(self.gpu_num):
                    if i < effective_gpu_num:
                        x, z_ = xs[i], z_s[i]
                        with tf.device('/device:GPU:%d' % i):
                            with tf.name_scope('%s_%d' % ('g', i)) as scope:
                                reuse = False if i == 0 else True
                                z, logit, y51 = light_model(x=x,
                                                            is_train=self.istraining,
                                                            rmax=self.rmax,
                                                            dmax=self.dmax,
                                                            ini=self.config,
                                                            reuse=reuse)

                                loss = self.loss(z=z, z_=z_)
                                self.losses.append(loss)
                                self.logits.append(logit)
                                self.y51s.append(y51)
                                tf.get_variable_scope().reuse_variables()
                                grads = TO.get_grads(optimizer=self.optimizer,
                                                     loss_function=loss,
                                                     var_list=var_list,
                                                     gradient_clipping=gradient_cliiping,
                                                     clipping_norm=clipping_norm,
                                                     clipping_type='norm')
                                # 精度の定義
                                accuracy = tf.sqrt(tf.reduce_mean(tf.multiply(
                                    tf.sigmoid(z) - z_, tf.sigmoid(z) - z_)))
                                with tf.device('/cpu:0'):
                                    vs.variable_summary(
                                        accuracy, 'Accuracy', is_scalar=True)

                                logger.debug(
                                    "04: TF Accuracy measure definition done")
                                tower_grads.append(grads)
                                logger.debug("03-05: Grads")
            grads = self.average_gradients(tower_grads)
            self.logit = Layers.concat(self.logits, concat_type='Batch')
            self.y51 = Layers.concat(self.y51s, concat_type='Batch')
            self.loss_function = tf.reduce_mean(self.losses)
            logger.debug("03-06: Average grads")
            self.train_op = TO.get_train_op(optimizer=self.optimizer,
                                            grad_var_pairs=grads,
                                            ema=False,
                                            ema_decay=0.9999)
            logger.debug("03-07: Train op")

    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def make_feed_dict(self, prob, data, label=None, is_Train=True, is_update=False, is_label=False):
        if self.steps <= 5000:
            rmax, dmax = 1.0, 0.0
        else:
            rmax = min(1.0 + 2.0 * float(self.steps - 5000.0) / 35000.0, 3.0)
            dmax = min(5.0 * float(self.steps - 5000.0) / 20000.0, 5.0)
        feed_dict = {}
        feed_dict.setdefault(self.x, data)
        if is_label:
            feed_dict.setdefault(self.z_, label)
        feed_dict.setdefault(self.learning_rate, self.learning_rate_value)
        feed_dict.setdefault(self.wd, self.wd_value)
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

    def cosine_decay(self):
        lr_min = self.config.getfloat('DLParams', 'learning_rate_min')
        lr_max = self.config.getfloat('DLParams', 'learning_rate')
        return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(self.t_cur * np.pi / self.t_i))

    def learning(self, data, save_at_log=False, validation_batch_num=1, batch_ratio=[0.2, 0.3, 0.4]):
        with tf.device('/cpu:0'):
            s = time.time()
            epoch = int(float(len(data.train.files)) *
                        float(self.epoch) / float(self.batch))
            one_epoch_step = int(
                float(len(data.train.files)) / float(self.batch))
            logger.debug("Step num: %d", epoch)
            for i in range(epoch):
                self.learning_rate_value = self.cosine_decay()
                br = np.random.randint(len(batch_ratio))
                batch = data.train.next_batch(
                    self.batch, batch_ratio=batch_ratio[br % len(batch_ratio)])
                # 途中経過のチェック
                if i % self.log == 0 and i != 0:
                    # Output
                    logger.debug(
                        "step %d / %d =================================================================================" % (i, epoch))
                    e = time.time()
                    elasped = e - s
                    logger.debug("elasped time: %g" % elasped)
                    s = e

                # バリデーションチェック
                if self.steps % self.dumping_period == 0 and self.steps != 0:
                    validation_data = data.val.get_all_files()
                    validation_loss = 0.0
                    for vnum in range(0, len(validation_data[0]), self.batch):
                        sp, ep = vnum, min(vnum + self.batch,
                                           len(validation_data[0]))
                        imgs = []
                        for f in validation_data[0][sp:ep]:
                            imgs.append(data.val.img_reader(f, False)[0])
                        l = [x for x in validation_data[2][sp:ep]]
                        feed_dict_val = self.make_feed_dict(
                            prob=True, data=np.array(imgs), label=np.array(l), is_Train=False, is_label=True)
                        v = self.sess.run([self.losse_function],
                                          feed_dict=feed_dict_val)
                        validation_loss += v[0] / \
                            float((len(validation_data[0]) // self.batch))
                    logger.debug("Before val: %g, After val: %g" %
                                 (self.prev_val, validation_loss))
                    self.prev_val = validation_loss
                    self.validation_save(str(int(validation_loss * 10000)))

                # 学習
                feed_dict = self.make_feed_dict(
                    prob=False, data=batch[0], label=batch[2], is_Train=True, is_update=True, is_label=True)
                self.sess.run([self.train_op], feed_dict=feed_dict)

                if i % self.tflog == 0 and i != 0:
                    feed_dict = self.make_feed_dict(
                        prob=False, data=batch[0], label=batch[2], is_Train=True, is_update=True, is_label=True)
                    summary = self.sess.run(self.summary, feed_dict=feed_dict)
                    vs.add_log(writer=self.train_writer,
                               summary=summary, step=self.steps)
                    validation_batch = data.val.next_batch(
                        self.batch, augment=False, batch_ratio=batch_ratio[br % len(batch_ratio)])
                    feed_dict_val = self.make_feed_dict(
                        prob=True, data=validation_batch[0], label=validation_batch[2], is_Train=False, is_label=True)
                    summary = self.sess.run(self.summary, feed_dict=feed_dict_val
                                            )
                    vs.add_log(writer=self.val_writer,
                               summary=summary, step=self.steps)
                    test_batch = data.test.next_batch(
                        self.batch, augment=False, batch_ratio=batch_ratio[br % len(batch_ratio)])
                    feed_dict_test = self.make_feed_dict(
                        prob=True, data=test_batch[0], label=test_batch[2], is_Train=False, is_label=True)
                    summary = self.sess.run(self.summary, feed_dict=feed_dict_test
                                            )
                    vs.add_log(writer=self.test_writer,
                               summary=summary, step=self.steps)
                self.steps += 1
                if self.steps % one_epoch_step == 0 and self.steps != 0:
                    self.t_cur += 1
                if self.t_cur > self.t_i:
                    self.t_i *= self.config.getfloat('DLParams', 't_mold')
                    self.t_cur = 0

            self.save_checkpoint()

    def get_roi_map_base(self, feed_dict):
        return self.sess.run([self.y51], feed_dict=feed_dict)

    def prediction(self, data, height=128, width=128, roi=False, label_def=None, save_dir=None,
                   filenames=None):
        # Make feed dict for prediction
        feed_dict = self.make_feed_dict(
            prob=True, data=data, is_Train=False, is_label=False)
        # Get logits
        result_z = self.sess.run(self.logit, feed_dict=feed_dict)
        result_y = [[1, 0] for i in range(len(result_z))]
        # Make ROI maps
        if not roi:
            return result_y, result_z, None, None
        else:
            roi_base = self.get_roi_map_base(feed_dict=feed_dict)
            result_roi = []
            for i in range(len(filenames)):
                roi_map, filepath = self.make_roi(roi_base=roi_base[0][i,
                                                                       :, :, :],
                                                  height=height,
                                                  width=width,
                                                  save_dir=save_dir,
                                                  filename=filenames[i],
                                                  label_def=label_def)
                result_roi.append(roi_map)
            return result_y, result_z, np.array(result_roi), filepath

    def make_roi(self, roi_base, height, width, save_dir, filename, label_def):
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
        roi_maps, filepath = [], []
        for x, finding in enumerate(label_def):
            # sum channels
            images = np.zeros((roi_base.shape[0], roi_base.shape[1], 3))
            c = roi_base[:, :, x]
            image = np.stack((c, c, c), axis=-1)
            images += image
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
            roi_img = cv2.resize(roi_img, (height, width))
            # save image
            basename = os.path.basename(filename)
            ftitle, _ = os.path.splitext(basename)
            unixtime = datetime.now().strftime('%s')
            cv2.imwrite(save_dir + '/' + unixtime + '_' + str(ftitle) +
                        '_' + str(finding) + '.png', roi_img)
            filepath.append(save_dir + '/' + str(ftitle) +
                            '_' + str(finding) + '.png')
        return np.array(roi_maps), filepath
