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

class DataSet(object):
    def __init__(self, Paths, Target, Kth = 3, Size = 512,
                 Augment = True):
        self.Size = Size
        self.Augmentation = Augment
        # ファイルパスの取得
        Files = []
        for Path in Paths:
            Files.extend(glob.glob(Path))
        Files = sorted(Files)
        # ターゲットファイルの抽出
        # ラベル別格納はTBD
        self.files = []
        for index, File in enumerate(Files):
            for number in Target:
                if int(index) % Kth == number - 1:
                    self.files.append(File)
        # ファイル配列のIDのリストを作成
        self.start = 0
        imgs, labels= [], []
        for i in range(len(self.files) - 1):
            imgs.append(i)
            labels.append(i)
        self._images = np.array(imgs)
        self._labels = np.array(labels)
        self.order_shuffle()


    def get_all_data(self):
        imgs, labels = [], []
        for i in range(len(self.files)):
            # ファイルの読み込み
            img, label = self.img_reader(self.files[i])
            # 出力配列の作成
            imgs.append(img), labels.append(label)
        return [np.array(imgs), np.array(labels, dtype = np.int32)]

    def img_reader(self, f):
        # 読み込み方式と教師ラベルの作成
        # PNG対応　
        root, ext = os.path.splitext(f)
        if ext == ".dcm":
            img, _ = dicom_to_np(f)
        elif ext == ".png":
            img = []
        else:
            img = []
        if f.find("BenchMark") >= 0:
            label = [1, 0] if root.find('JPCLN') >= 0 else [0, 1]
        elif f.find("AdditialnalData") >= 0:
            label = [1, 0]
        else:
            label = [0, 0]
        # 画像サイズの調整
        img = cv2.resize(img,(self.Size,self.Size), interpolation = cv2.INTER_AREA)
        # ZCA whitening
        img = PP.PreProcessing(np.reshape(img, (self.Size,self.Size, 1)))
        # データオーギュメンテーション
        if self.Augmentation:
            img = self.flip(img)
            img = self.shift(img = img, move_x = 0.05, move_y = 0.05)
        return img, label

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


    def order_shuffle(self):
        perm = np.arange(len(self._images))
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

    def next_batch(self, batch_size):
        start = self.start
        if self.start + batch_size >= len(self._images):
            print("Next epoch")
            self.order_shuffle()
            start = 0
            end = batch_size
        else:
            end = min(self.start + batch_size, len(self._images) - 1)
        self.start = end
        imgs, labels = [], []
        for i in range(start, end):
            # ファイルの読み込み
            img, label = self.img_reader(self.files[self._images[i]])
            # 出力配列の作成
            imgs.append(img), labels.append(label)
        return [np.array(imgs), np.array(labels, dtype = np.int32)]

def read_data_sets(Paths, Train = [1, 2], Test = [3], Size = 512,
                   Augment = True):
    class DataSets(object):
        pass
    Kth = len(Train) + len(Test)
    data_sets = DataSets()
    data_sets.train = DataSet(Paths = Paths, Target = Train, Kth = Kth, Size = Size,
                              Augment = Augment)
    data_sets.test = DataSet(Paths = Paths, Target = Test, Kth = Kth, Size = Size,
                             Augment = Augment)

    return data_sets

if __name__ == '__main__':
    data = read_data_sets(Paths = ["./Data/CR_DATA/BenchMark/*/*.dcm",
                                   "./Data/CR_DATA/AdditionalData/*.dcm"])
    print(len(data.train.get_all_data()[1]))
    print(len(data.test.get_all_data()[1]))
    for i in range(2):
        print(data.train.next_batch(10))
