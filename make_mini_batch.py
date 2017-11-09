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
    def __init__(self,
                 data, label,
                 size,
                 zca,
                 augment):
        pass


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
        findings.setdefault(p, {'label' : np.array([label0, label1])})
    print(findings)
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

    return data_sets

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
