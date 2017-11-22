#! /usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function
import glob
import os
import random
import csv
import numpy as np
import cv2
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
                 augment):
        self.size = size
        self.augment = augment
        self.zca = zca
        self.files = data
        self.labels = label
        # ファイル配列のIDのリストを作成
        self.start = 0
        imgs, labels= [], []
        logger.debug("File num: %g"%len(self.files))
        for i in range(len(self.files)):
            imgs.append(i)
            labels.append(i)
        self._images = np.array(imgs)
        self._labels = np.array(labels)
        self.order_shuffle()

    def order_shuffle(self):
        perm = np.arange(len(self._images))
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

    def flip(self,img):
        if random.random() >= 0.8:
            img = cv2.flip(img, 0)
        if random.random() >= 0.8:
            img = cv2.flip(img, 1)
        img = img.reshape((img.shape[0], img.shape[1], 1))
        return img

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
            img = img.reshape((img.shape[0], img.shape[1], 1))
            return img
        else:
            return img

    def get_all_data(self):
        imgs, labels0, labels1 = [], [], []
        filenames, row_data = [], []
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
            img, _ = dicom_to_np(f)
        elif ext == ".png":
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        else:
            img = []

        # 教師データの読み込み
        label = self.labels[filename]['label']

        # 画像サイズの調整
        img = cv2.resize(img,(self.size,self.size), interpolation = cv2.INTER_AREA)
        # ZCA whitening
        if self.zca:
            img = PP.PreProcessing(np.reshape(img, (self.size,self.size, 1)))
        else:
            img = np.reshape(img, (self.size,self.size, 1))
        # データオーギュメンテーション
        if self.augment:
            img = self.flip(img)
            img = self.shift(img = img, move_x = 0.05, move_y = 0.05)
        else:
            if augment:
                img = self.flip(img)
                img = self.shift(img = img, move_x = 0.05, move_y = 0.05)


        return img, label[0], label[1], filename, self.labels[filename]['raw']

    def next_batch(self, batch_size, augment = True):
        start = self.start
        if self.start + batch_size >= len(self._images):
            logger.debug('Next Epoch')
            shuffle = True
        else:
            shuffle = False
        end = min(self.start + batch_size, len(self._images) - 1)

        imgs, labels0, labels1 = [], [], []
        filenames, raw_data = [], []
        for i in range(start, end):
            # ファイルの読み込み
            img, label0, label1, filename, raw = self.img_reader(self.files[self._images[i]],
                                                                 augment = augment)
            # 出力配列の作成
            imgs.append(img)
            labels0.append(label0)
            labels1.append(label1)
            filenames.append(filename)
            raw_data.append(raw)
        if shuffle:
            self.order_shuffle()
            self.start = 0
        else:
            self.start = end

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

def make_supevised_data_for_conf(path, labels):
    findings = {}
    for p in path:
        label0 = np.zeros(len(labels))
        label1 = np.zeros(2)
        if p.find('JPCNN') >= 0:
            label1[0] = 1
        else:
            label1[1] = 1
        findings.setdefault(os.path.basename(p),
                            {'label' : np.array([label0, label1]),
                             'raw' : p})
    return findings

def read_data_sets(nih_datapath = ["./Data/Open/images/*.png"],
                   nih_supervised_datapath = "./Data/Open/Data_Entry_2017.csv",
                   nih_boxlist = "./Data/Open/BBox_List_2017.csv",
                   benchmark_datapath = ["./Data/CR_DATA/BenchMark/*/*.dcm"],
                   benchmark_supervised_datapath = "./Data/CR_DATA/BenchMark/CLNDAT_EN.txt",
                   kfold = 1,
                   img_size = 512,
                   augment = True,
                   zca = True):
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
                                               label_def)

    data_sets.train = DataSet(data = nih_data,
                              label = nih_labels,
                              size = img_size,
                              zca = zca,
                              augment = augment)
    data_sets.test  = DataSet(data = conf_data,
                              label = conf_labels,
                              size = img_size,
                              zca = zca,
                              augment = augment)
    data_sets.train_summary = nih_count
    return data_sets, label_def

if __name__ == '__main__':
    dataset = read_data_sets(nih_datapath = ["./Data/Open/images/*.png"],
                             nih_supervised_datapath = "./Data/Open/Data_Entry_2017.csv",
                             nih_boxlist = "./Data/Open/BBox_List_2017.csv",
                             benchmark_datapath = ["./Data/CR_DATA/BenchMark/*/*.dcm"],
                             benchmark_supervised_datapath = "./Data/CR_DATA/BenchMark/CLNDAT_EN.txt",
                             kfold = 1,
                             img_size = 512,
                             augment = True,
                             zca = True)
    print(len(dataset.test.get_all_data()), len(dataset.test.get_all_data()[2]))
    for i in range(2):
        x = dataset.train.next_batch(1)
        print(x[1], x[2], x[3], x[4])
        y = dataset.test.next_batch(2)
        print(y[1], y[2], y[3], y[4])
    for i in tqdm(range(100)):
        y = dataset.test.next_batch(20)
