#!/usr/bin/env python
# -*- coding: utf-8 -*-
from make_mini_batch import read_data_sets
from detection import Detecter
from logging import getLogger, StreamHandler
from tqdm import tqdm
import os
import csv
import argparse
import cv2
from utils import *
import numpy as np
import json
import tensorflow as tf
from DICOMReader.DICOMReader import dicom_to_np
from preprocessing_tool import preprocessing as PP
logger = getLogger(__name__)
sh = StreamHandler()
logger.addHandler(sh)
logger.setLevel(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config')
    parser.add_argument('-file')
    parser.add_argument('-dir')
    args = parser.parse_args()
    filename = args.file
    dirname = args.dir

    size, augment, checkpoint, lr, dlr, rtype, rr, l1_norm, dumping_rate, dumping_period, epoch, batch, log, tflog, ds, roi, output_type, outfile, mode, step, split_mode, network_mode, auc_file, validation_set, optimizer_type, config = config_list(
        args)

    if mode in ['learning']:
        init = True
    elif mode in ['update', 'prediction']:
        init = False
    else:
        init = False

    obj = Detecter(output_type=output_type,
                   epoch=epoch, batch=batch, log=log,
                   optimizer_type='Adam',
                   learning_rate=lr,
                   dynamic_learning_rate=dlr,
                   beta1=0.9, beta2=0.999,
                   dumping_period=dumping_period,
                   dumping_rate=dumping_rate,
                   regularization=rr,
                   regularization_type=rtype,
                   checkpoint=checkpoint,
                   init=init,
                   size=size,
                   l1_norm=l1_norm,
                   step=step,
                   network_mode=network_mode,
                   tflog=tflog)
    obj.construct()
    label_list = json.load(open('./Config/label_def.json'))
    root, ext = os.path.splitext(filename)
    img = img_process(f=filename, ext=ext, size=size, model='densenet')
    ts = [img]
    x, y, z, filepath = obj.prediction(data=ts, roi=roi,
                                       label_def=label_list['label_def'], save_dir=dirname,
                                       filenames=[filename],
                                       suffixs=['result'],
                                       roi_force=True)
    print("File name:", filename)
    print(y[0])
    s = [0 for j in range(len(y[0]))]
    for i, diag in enumerate(label_list['label_def']):
        roc_map = np.load('./Config/' + diag + '.npy')
        for line in roc_map:
            if y[0][i] <= float(line[2]):
                s[i] = float(line[0])
    print("Fibrosis probability:", s[6])
    print("Fibrosis filepath:", filepath[6])
    return s[6], filepath[6]


def img_process(f, ext, size, model):
    # 画像の読み込み
    if ext == ".dcm":
        img, bits = dicom_to_np(f)
        img = 255.0 * img / bits
        img = img.astype(np.uint8)
    elif ext == ".png":
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    else:
        img = []

    if model == 'xception':
        pi = tf.keras.applications.xception.preprocess_input
    elif model == 'resnet':
        pi = tf.keras.applications.resnet50.preprocess_input
    elif model == 'inception':
        pi = tf.keras.applications.inception_v3.preprocess_input
    elif model == 'densenet':
        pi = tf.keras.applications.densenet.preprocess_input
    else:
        pi = tf.keras.applications.vgg19.preprocess_input

    # 画像サイズの調整
    img = cv2.resize(img, (size, size),
                     interpolation=cv2.INTER_AREA)

    img = (img.astype(np.int32)).astype(np.float32)
    img = np.stack((img, img, img), axis=-1)
    img = pi(img.astype(np.float32))
    return img


if __name__ == '__main__':
    main()
