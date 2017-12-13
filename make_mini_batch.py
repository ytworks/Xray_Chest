#! /usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function
import glob
import os
import random
import csv
import numpy as np
import cv2
import time
import tensorflow as tf
from DICOMReader.DICOMReader import dicom_to_np
from preprocessing_tool import preprocessing as PP
from tqdm import tqdm
from logging import getLogger, StreamHandler
logger = getLogger(__name__)
sh = StreamHandler()
logger.addHandler(sh)
logger.setLevel(10)


class DataSet(object):
    def __init__(self,
                 data, label,
                 size,
                 zca,
                 augment,
                 raw_img,
                 model):
        self.size = size
        self.augment = augment
        self.zca = zca
        self.files = data
        self.labels = label
        self.raw_img = raw_img
        self.channel = 1 if not self.raw_img else 3
        logger.debug("Channel %s" % str(self.channel))
        logger.debug("Size %s" % str(self.size))
        if model == 'xception':
            self.pi = tf.keras.applications.xception.preprocess_input
        elif model == 'resnet':
            self.pi = tf.keras.applications.resnet50.preprocess_input
        elif model == 'inception':
            self.pi = tf.keras.applications.inception_v3.preprocess_input
        else:
            self.pi = tf.keras.applications.vgg19.preprocess_input

        # 正常/異常のファイルの分類
        self.normal, self.abnormal = [], []
        for filename in self.files:
            base_filename = os.path.basename(filename)
            if self.labels[base_filename]['label'][1][0] == 1:
                self.normal.append(filename)
            else:
                self.abnormal.append(filename)


        # ファイル配列のIDのリストを作成
        self.start_normal, self.start_abnormal = 0, 0
        imgs_normal, imgs_abnormal= [], []
        logger.debug("File num: %g"%len(self.files))
        logger.debug("Normal File num: %g"%len(self.normal))
        logger.debug("Abnormal File num: %g"%len(self.abnormal))
        for i in range(len(self.normal)):
            imgs_normal.append(i)
        for i in range(len(self.abnormal)):
            imgs_abnormal.append(i)
        self._images_normal = np.array(imgs_normal)
        self._images_abnormal = np.array(imgs_abnormal)
        self.order_shuffle_normal()
        self.order_shuffle_abnormal()

    def order_shuffle_normal(self):
        perm = np.arange(len(self._images_normal))
        np.random.shuffle(perm)
        self._images_normal = self._images_normal[perm]

    def order_shuffle_abnormal(self):
        perm = np.arange(len(self._images_abnormal))
        np.random.shuffle(perm)
        self._images_abnormal = self._images_abnormal[perm]

    def flip(self,img):
        #if random.random() >= 0.8:
        #    img = cv2.flip(img, 0)
        if random.random() >= 0.8:
            img = cv2.flip(img, 1)
        if random.random() >= 0.8:
            img = self.rotation(img, rot = random.choice([0, 90, 180, 270]))
        img = img.reshape((img.shape[0], img.shape[1], self.channel))
        return img

    def rotation(self, img, rot = 45):
        size = tuple(np.array([img.shape[1], img.shape[0]]))
        matrix = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),rot,self.channel)
        affine_matrix = np.float32(matrix)
        return cv2.warpAffine(img, affine_matrix, size, flags=cv2.INTER_LINEAR)

    def shift(self, img, move_x = 0.1, move_y = 0.1):
        if random.random() >= 0.8:
            size = tuple(np.array([img.shape[0], img.shape[1]]))
            mx = int(img.shape[0] * move_x * random.random())
            my = int(img.shape[1] * move_y * random.random())
            matrix = [
                        [1,  0, mx],
                        [0,  1, my]
                    ]
            affine_matrix = np.float32(matrix)
            img = cv2.warpAffine(img, affine_matrix, size, flags=cv2.INTER_LINEAR)
            img = img.reshape((img.shape[0], img.shape[1], self.channel))
            return img
        else:
            return img

    def get_all_data(self):
        imgs, labels0, labels1 = [], [], []
        filenames, raw_data = [], []
        for i in tqdm(range(len(self.files))):
            # ファイルの読み込み
            img, label0, label1, filename, raw = self.img_reader(self.files[i], augment = False)
            # 出力配列の作成
            imgs.append(img)
            labels0.append(label0)
            labels1.append(label1)
            filenames.append(filename)
            raw_data.append(raw)
        return [np.array(imgs), np.array(labels1), np.array(labels0), filenames, raw_data]

    def img_reader(self, f, augment = True):
        root, ext = os.path.splitext(f)
        filename = os.path.basename(f)
        # 画像の読み込み
        if ext == ".dcm":
            img, bits = dicom_to_np(f)
            if ext == ".dcm":
                img = 255.0 * img / bits
        elif ext == ".png":
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        else:
            img = []

        # 教師データの読み込み
        label = self.labels[filename]['label']

        # 画像サイズの調整
        img = cv2.resize(img,(self.size,self.size), interpolation = cv2.INTER_AREA)
        # ZCA whitening
        if not self.raw_img:
            if self.zca:
                img = PP.PreProcessing(np.reshape(img, (self.size,self.size, self.channel)))
            else:
                img = np.reshape(img, (self.size,self.size, self.channel))
        else:
            img = (img.astype(np.int32)).astype(np.float32)
            img = np.stack((img, img, img), axis = -1)
            img = self.pi(img)
        # データオーギュメンテーション
        if self.augment:
            img = self.flip(img)
            #img = self.shift(img = img, move_x = 0.05, move_y = 0.05)
        else:
            if augment:
                img = self.flip(img)
                #img = self.shift(img = img, move_x = 0.05, move_y = 0.05)


        return img, label[0], label[1], filename, self.labels[filename]['raw']

    def next_batch(self, batch_size, augment = True, debug = True, batch_ratio = 0.5):
        # 正常系の制御
        start_normal = self.start_normal
        if self.start_normal + int(batch_size * batch_ratio) >= len(self._images_normal):
            logger.debug('Normal Next Epoch')
            shuffle_normal = True
        else:
            shuffle_normal = False
        end_normal = min(self.start_normal + int(batch_size * batch_ratio), len(self._images_normal) - 1)

        # 異常系の制御
        start_abnormal = self.start_abnormal
        if self.start_abnormal + int(batch_size * batch_ratio) >= len(self._images_abnormal):
            logger.debug('Abnormal Next Epoch')
            shuffle_abnormal = True
        else:
            shuffle_abnormal = False
        end_abnormal = min(self.start_abnormal + int(batch_size * batch_ratio), len(self._images_abnormal) - 1)


        imgs, labels0, labels1 = [], [], []
        filenames, raw_data = [], []
        # 正常系
        for i in range(start_normal, end_normal):
            # ファイルの読み込み
            img, label0, label1, filename, raw = self.img_reader(self.normal[self._images_normal[i]],
                                                                 augment = augment)
            # 出力配列の作成
            imgs.append(img)
            labels0.append(label0)
            labels1.append(label1)
            filenames.append(filename)
            raw_data.append(raw)

        # 異常系
        for i in range(start_abnormal, end_abnormal):
            # ファイルの読み込み
            img, label0, label1, filename, raw = self.img_reader(self.abnormal[self._images_abnormal[i]],
                                                                 augment = augment)
            # 出力配列の作成
            imgs.append(img)
            labels0.append(label0)
            labels1.append(label1)
            filenames.append(filename)
            raw_data.append(raw)

        # 正常系シャッフル
        if shuffle_normal:
            self.order_shuffle_normal()
            self.start_normal = 0
        else:
            self.start_normal = end_normal

        # 異常系シャッフル
        if shuffle_abnormal:
            self.order_shuffle_abnormal()
            self.start_abnormal = 0
        else:
            self.start_abnormal = end_abnormal



        return [np.array(imgs), np.array(labels1), np.array(labels0), filenames, raw_data]


def get_filepath(datapaths):
    files = []
    for path in datapaths:
        files.extend(glob.glob(path))
    files = sorted(files)
    return files


def make_supevised_data_for_nih(path):
    findings = {}
    # ファイルの読み込み
    with open(path, 'rU') as f:
        lines = csv.reader(f)
        lines.next()
        for line in lines:
            findings.setdefault(line[0], {'raw' : line[1]})
    logger.debug('NIH # of Data records: %d'%len(findings))
    # データ数のカウント
    finding_count = {}
    for file_name, finding in findings.items():
        for f in finding['raw'].split('|'):
            finding_count.setdefault(f, 0)
            finding_count[f] += 1
    # バイナライズ
    binary_def = [l for l in sorted(finding_count.keys()) if not l in 'No Finding']
    for file_name, finding in findings.items():
        label0 = np.zeros(len(binary_def))
        label1 = np.zeros(2)
        for i, b in enumerate(binary_def):
            if finding['raw'].find(b) >= 0:
                label0[i] = 1
            if finding['raw'].find('No Finding') >= 0:
                label1[0] = 1
            else:
                label1[1] = 1
        findings[file_name].setdefault('label', np.array([label0, label1]))
    return findings, finding_count, binary_def

def make_supevised_data_for_conf(path, labels, datapath):
    reader = csv.reader(open(datapath, 'rU'), delimiter = '\t')
    img2diag = {}
    for row in reader:
        if row != []:
            img2diag.setdefault(row[0].replace('.IMG', ''), row[10]+row[11])
    diags = list(set([v for v in img2diag.values()]))
    mapper = diagnosis_map(diags, labels)
    findings = {}
    for p in path:
        label0 = np.zeros(len(labels))
        label1 = np.zeros(2)
        if p.find('JPCNN') >= 0:
            label1[0] = 1
        else:
            label1[1] = 1
            file_name, _ = os.path.splitext(os.path.basename(p))
            for index in mapper[img2diag[file_name]]:
                    label0[index] = 1
        findings.setdefault(os.path.basename(p),
                            {'label' : np.array([label0, label1]),
                             'raw' : p})
    return findings

def diagnosis_map(diags, labels):
    mapper = {}
    for d in diags:
        mapper.setdefault(d, [])
        for i, l in enumerate(labels):
            if l in ['Mass', 'Nodule']:
                mapper[d].append(i)
            if d.find('pneumonia') >= 0 and l == 'Pneumonia':
                mapper[d].append(i)
    return mapper



def read_data_sets(nih_datapath = ["./Data/Open/images/*.png"],
                   nih_supervised_datapath = "./Data/Open/Data_Entry_2017.csv",
                   nih_boxlist = "./Data/Open/BBox_List_2017.csv",
                   benchmark_datapath = ["./Data/CR_DATA/BenchMark/*/*.dcm"],
                   benchmark_supervised_datapath = "./Data/CR_DATA/BenchMark/CLNDAT_EN.txt",
                   kfold = 1,
                   img_size = 512,
                   augment = True,
                   zca = True,
                   raw_img = False,
                   model = 'xception'):
    class DataSets(object):
        pass
    data_sets = DataSets()

    # NIHのデータセットのファイルパスを読み込む
    nih_data = get_filepath(nih_datapath)

    # 学会データセットのファイルパスを読み込む
    conf_data = get_filepath(benchmark_datapath)

    # NIHの教師データを読み込む
    nih_labels, nih_count, label_def = make_supevised_data_for_nih(nih_supervised_datapath)

    # 学会データの教師データを読み込む
    conf_labels = make_supevised_data_for_conf(conf_data,
                                               label_def,
                                               benchmark_supervised_datapath)

    data_sets.train = DataSet(data = nih_data,
                              label = nih_labels,
                              size = img_size,
                              zca = zca,
                              augment = augment,
                              raw_img = raw_img,
                              model = model)
    data_sets.test  = DataSet(data = conf_data,
                              label = conf_labels,
                              size = img_size,
                              zca = zca,
                              augment = augment,
                              raw_img = raw_img,
                              model = model)
    data_sets.train_summary = nih_count
    return data_sets, label_def

if __name__ == '__main__':
    # raw_img
    dataset, _ = read_data_sets(nih_datapath = ["./Data/Open/images/*.png"],
                             nih_supervised_datapath = "./Data/Open/Data_Entry_2017.csv",
                             nih_boxlist = "./Data/Open/BBox_List_2017.csv",
                             benchmark_datapath = ["./Data/CR_DATA/BenchMark/*/*.dcm"],
                             benchmark_supervised_datapath = "./Data/CR_DATA/BenchMark/CLNDAT_EN.txt",
                             kfold = 1,
                             img_size = 512,
                             augment = True,
                             zca = True,
                             raw_img = True,
                             model = 'inception')
    print(len(dataset.test.get_all_data()), len(dataset.test.get_all_data()[2]))
    for i in range(2):
        x = dataset.train.next_batch(4)
        print(x[1], x[2], x[3], x[4])
        y = dataset.test.next_batch(6)
        print(y[1], y[2], y[3], y[4])
    for i in tqdm(range(100)):
        y = dataset.test.next_batch(20)

    # raw_img
    dataset, _ = read_data_sets(nih_datapath = ["./Data/Open/images/*.png"],
                             nih_supervised_datapath = "./Data/Open/Data_Entry_2017.csv",
                             nih_boxlist = "./Data/Open/BBox_List_2017.csv",
                             benchmark_datapath = ["./Data/CR_DATA/BenchMark/*/*.dcm"],
                             benchmark_supervised_datapath = "./Data/CR_DATA/BenchMark/CLNDAT_EN.txt",
                             kfold = 1,
                             img_size = 512,
                             augment = True,
                             zca = True,
                             raw_img = False)
    print(len(dataset.test.get_all_data()), len(dataset.test.get_all_data()[2]))
    for i in range(2):
        x = dataset.train.next_batch(4)
        print(x[1], x[2], x[3], x[4])
        y = dataset.test.next_batch(6)
        print(y[1], y[2], y[3], y[4])
    for i in tqdm(range(100)):
        y = dataset.test.next_batch(20)
