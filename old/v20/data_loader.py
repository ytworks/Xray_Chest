#! /usr/bin/env python
# coding:utf-8
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


def get_filepath(datapaths, filter_list = None):
    files = []
    for path in datapaths:
        files.extend(glob.glob(path))
    files = sorted(files)
    if filter_list == None:
        return files
    else:
        filter_files = []
        for filename in files:
            base_filename = os.path.basename(filename)
            if filter_list.has_key(base_filename):
                filter_files.append(filename)
        return filter_files


def make_supevised_data_for_nih(path, filter_list = None):
    # 有効ファイルの読み込み
    valid_files = {}
    if filter_list != None:
        with open(filter_list, 'r') as f:
            lines = csv.reader(f)
            for line in lines:
                valid_files.setdefault(line[0], True)

    findings = {}
    # ファイルの読み込み
    with open(path, 'rU') as f:
        lines = csv.reader(f)
        lines.next()
        for line in lines:
            if filter_list == None or valid_files.has_key(line[0]):
                findings.setdefault(line[0], {'raw' : line[1]})
    logger.debug('NIH # of Data records: %d'%len(findings))
    # データ数のカウント
    finding_count = {}
    for file_name, finding in findings.items():
        for f in finding['raw'].split('|'):
            finding_count.setdefault(f, 0)
            finding_count[f] += 1
    # バイナライズ
    #binary_def = [l for l in sorted(finding_count.keys()) if not l in 'No Finding']
    binary_def = [l for l in sorted(finding_count.keys())]
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
    no_findings = [i for i, label in enumerate(labels) if label in 'No Finding']
    findings = {}
    for p in path:
        label0 = np.zeros(len(labels))
        label1 = np.zeros(2)
        if p.find('JPCNN') >= 0:
            label1[0] = 1
            label0[no_findings[0]] = 1
        else:
            label1[1] = 1
            file_name, _ = os.path.splitext(os.path.basename(p))
            for index in mapper[img2diag[file_name]]:
                    label0[index] = 1
        findings.setdefault(os.path.basename(p),
                            {'label' : np.array([label0, label1]),
                             'raw' : p,
                             'filepath' : p})
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


def merge_filepath(filepaths, labels):
    exist_labels = {}
    for filepath in filepaths:
        basename = os.path.basename(filepath)
        labels[basename].setdefault('filepath', filepath)
        exist_labels.setdefault(basename, {})
        exist_labels[basename].setdefault('raw', labels[basename]['raw'])
        exist_labels[basename].setdefault('label', labels[basename]['label'])
        exist_labels[basename].setdefault('filepath', filepath)
    return exist_labels


class DataSet(object):
    def __init__(self,
                 data,
                 size,
                 augment,
                 model,
                 raw_img,
                 is_train, batch_size):
        self.size = size
        self.augment = augment
        self.files = data
        self.data = data
        self.raw_img = raw_img
        self.channel = 1 if not self.raw_img else 3
        self.is_train = is_train
        self.batch_size = batch_size
        logger.debug("Channel %s" % str(self.channel))
        logger.debug("Size %s" % str(self.size))
        logger.debug("Augmentation %s" % str(self.augment))
        logger.debug("Is train %s" % str(self.is_train))
        if model == 'xception':
            self.pi = tf.keras.applications.xception.preprocess_input
        elif model == 'resnet':
            self.pi = tf.keras.applications.resnet50.preprocess_input
        elif model == 'inception':
            self.pi = tf.keras.applications.inception_v3.preprocess_input
        elif model == 'densenet':
            self.pi = tf.keras.applications.densenet.preprocess_input
        elif model == 'ZCA':
            self.pi = PP.PreProcessing
        else:
            self.pi = tf.keras.applications.vgg19.preprocess_input

        self._make_features_and_labels()
        self._make_batch()

    def _make_features_and_labels(self):
        self.filenames, self.labels = [], []
        for k, v in self.data.items():
            self.filenames.append(v['filepath'])
            self.labels.append(tf.constant(v['label'][0]))
        print(len(self.labels), len(self.filenames))

    def _make_batch(self):
        self.filenames = tf.constant(self.filenames)

        self.dataset = tf.data.Dataset.from_tensor_slices((self.filenames, self.labels))
        self.dataset = self.dataset.map(lambda filename, label :
                                        tuple(tf.py_func(self.load_file, [filename, label],
                                                         [tf.float32, label.dtype])))
        self.dataset = self.dataset.shuffle(buffer_size=10000)
        self.dataset = self.dataset.batch(self.batch_size)
        self.dataset = self.dataset.repeat()
        self.iterator = self.dataset.make_one_shot_iterator()

    def load_file(self, filename, label):
        root, ext = os.path.splitext(filename)
        # 画像の読み込み
        if ext == ".dcm":
            img, bits = dicom_to_np(filename)
            if self.raw_img or self.zca == False:
                img = 255.0 * img / bits
                img = img.astype(np.uint8)
        elif ext == ".png":
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        else:
            img = []

        #img = (img.astype(np.int32)).astype(np.float32)

        # augmentation
        img = self.augmentation(img)
        # resize
        img, label = self.resize(img, label)
        img = self.pi(img.astype(np.float32))



        return img.astype(np.float32), label

    def resize(self, img, label):
        img = cv2.resize(img,(self.size,self.size), interpolation = cv2.INTER_AREA)
        img = np.stack((img, img, img), axis = -1)
        return img, label

    def augmentation(self, img):
        if self.augment and self.is_train:
            # zoom
            img = self.zoom(img)
            # flip
            img = self.flip(img = img)
            # rotation
            #if np.random.rand() >= 0.9:
            #    img = self.rotation(img, rot = random.choice([0, 90, 180, 270]))
            #    img = img.reshape((img.shape[0], img.shape[1], 1))
            # Shift
            img = self.shift(img = img, move_x = 0.05, move_y = 0.05)
            # small rotation
            if np.random.rand() >= 0.7:
                img = self.rotation(img, rot = 15.0 * (2.0 * random.random() - 1.0))
                img = img.reshape((img.shape[0], img.shape[1], 1))
        return img

    def zoom(self, img):
        if np.random.rand() >= 0.5:
            return img
        else:
            w, h = img.shape[0], img.shape[1]
            w_crop, h_crop = int(w * 0.1), int(h * 0.1)
            crop_size = min(w, h) - np.random.randint(1, min(w_crop, h_crop))
            start_w = np.random.randint(w - crop_size)
            start_h = np.random.randint(h - crop_size)
            end_w = start_w + crop_size
            end_h = start_h + crop_size
            img = img[start_w:end_w, start_h:end_h]
            return img



    def flip(self,img):
        #if np.random.rand() >= 0.9:
        #    img = cv2.flip(img, 0)
        #    img = img.reshape((img.shape[0], img.shape[1], 1))
        if np.random.rand() >= 0.7:
            img = cv2.flip(img, 1)
            img = img.reshape((img.shape[0], img.shape[1], 1))
        return img

    def rotation(self, img, rot = 45):
        size = tuple(np.array([img.shape[1], img.shape[0]]))
        matrix = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),rot,1)
        affine_matrix = np.float32(matrix)
        return cv2.warpAffine(img, affine_matrix, size, flags=cv2.INTER_LINEAR)

    def shift(self, img, move_x = 0.1, move_y = 0.1):
        if np.random.rand() >= 0.7:
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

    def get_iter(self):
        return self.iterator


def read_data_sets(nih_datapath = ["./Data/Open/images/*.png"],
                   nih_supervised_datapath = "./Data/Open/Data_Entry_2017.csv",
                   nih_boxlist = "./Data/Open/BBox_List_2017.csv",
                   nih_train_list = "./Data/Open/train_val_list.txt",
                   nih_test_list = "./Data/Open/test_list.txt",
                   benchmark_datapath = ["./Data/CR_DATA/BenchMark/*/*.dcm"],
                   benchmark_supervised_datapath = "./Data/CR_DATA/BenchMark/CLNDAT_EN.txt",
                   kfold = 1,
                   img_size = 512,
                   augment = True,
                   zca = True,
                   raw_img = False,
                   model = 'xception',
                   ds = 'normal',
                   batch_size = 16):
    class DataSets(object):
        pass
    data_sets = DataSets()

    # NIHの教師データを全て読み込む
    nih_labels, nih_count, label_def = make_supevised_data_for_nih(nih_supervised_datapath)
    # NIHのデータセットのファイルパスを読み込む
    nih_data = get_filepath(nih_datapath)

    # NIHの教師データをトレーニングセットのみ読み込む
    nih_labels_train, nih_count_train, label_def_train = make_supevised_data_for_nih(nih_supervised_datapath,
                                                                                     nih_train_list)
    # NIHの教師データセットのファイルパスを読み込む
    nih_data_train = get_filepath(nih_datapath, nih_labels_train)
    train_labels = merge_filepath(filepaths = nih_data_train, labels = nih_labels_train)

    # NIHのテストデータをテストセットのみ読み込む
    nih_labels_test, nih_count_test, label_def_test = make_supevised_data_for_nih(nih_supervised_datapath,
                                                                                  nih_test_list)
    # NIHのテストデータセットのファイルパスを読み込む
    nih_data_test = get_filepath(nih_datapath, nih_labels_test)
    test_labels = merge_filepath(filepaths = nih_data_test, labels = nih_labels_test)

    # 学会データセットのファイルパスを読み込む
    conf_data = get_filepath(benchmark_datapath)

    # 学会データの教師データを読み込む
    conf_labels = make_supevised_data_for_conf(conf_data,
                                               label_def,
                                               benchmark_supervised_datapath)
    logger.debug("Train")
    data_sets.train = DataSet(data = train_labels,
                              size = img_size,
                              augment = augment,
                              model = model,
                              raw_img = raw_img,
                              is_train = True,
                              batch_size = batch_size)
    logger.debug("Test")
    data_sets.test  = DataSet(data = test_labels,
                              size = img_size,
                              augment = augment,
                              model = model,
                              raw_img = raw_img,
                              is_train = False,
                              batch_size = batch_size)
    logger.debug("Conf")
    data_sets.conf  = DataSet(data = conf_labels,
                              size = img_size,
                              augment = augment,
                              model = model,
                              raw_img = raw_img,
                              is_train = False,
                              batch_size = batch_size)
    data_sets.train_summary = nih_count

    return data_sets, label_def

if __name__ == '__main__':
    n = 16
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
                             model = 'densenet',
                             batch_size = n)

    sess = tf.InteractiveSession()
    features, labels = dataset.train.get_iter().get_next()
    x, y = sess.run([features, labels])
    print(x.shape)
    for i in range(n):
        cv2.imshow('window',x[i])
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        print(x[i].max(), x[i].min(), x[i].mean())
    features, labels = dataset.conf.get_iter().get_next()
    x, y = sess.run([features, labels])
    for i in range(n):
        cv2.imshow('window', x[i])
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        print(x[i].max(), x[i].min(), x[i].mean())
