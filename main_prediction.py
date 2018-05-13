#!/usr/bin/env python
# -*- coding: utf-8 -*-
from make_mini_batch import read_data_sets
from detection import Detecter
from logging import getLogger, StreamHandler
from tqdm import tqdm
import os
import csv
import argparse
import ConfigParser as cp
import cv2
from utils import *
logger = getLogger(__name__)
sh = StreamHandler()
logger.addHandler(sh)
logger.setLevel(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config')
    parser.add_argument('-file')
    args = parser.parse_args()
    filename = args.file
    print(filename)


    config = cp.SafeConfigParser()
    config.read(args.config)
    show_config(config)
    size = config.getint('DLParams', 'size')
    augment = config.getboolean('DLParams', 'augmentation')
    checkpoint = config.get('OutputParams', 'checkpoint')
    lr = config.getfloat('DLParams', 'learning_rate')
    dlr = config.getfloat('DLParams', 'dynamic_learning_rate')
    rtype = config.get('DLParams', 'regularization_type')
    rr = config.getfloat('DLParams', 'regularization_rate')
    l1_norm = config.getfloat('DLParams', 'l1_normalization')
    epoch = config.getint('DLParams', 'epoch')
    batch = config.getint('DLParams', 'batch')
    log = config.getint('LogParams', 'log_period')
    ds = config.get('InputParams', 'dataset')
    roi = config.getboolean('Mode', 'roi_prediction')
    output_type = config.get('DLParams', 'output_type')
    outfile = config.get('OutputParams', 'outfile')
    mode = config.get('Mode', 'running_mode')
    step = config.getint('DLParams', 'step')
    split_mode = config.get('Mode', 'split_mode')
    network_mode = config.get('Mode', 'network_mode')
    auc_file = config.get('OutputParams', 'auc_file')


    dataset, label_def = read_data_sets(nih_datapath = ["./Data/Open/images/*.png"],
                             nih_supervised_datapath = "./Data/Open/Data_Entry_2017_v2.csv",
                             nih_boxlist = "./Data/Open/BBox_List_2017.csv",
                             benchmark_datapath = ["./Data/CR_DATA/BenchMark/*/*.dcm"],
                             benchmark_supervised_datapath = "./Data/CR_DATA/BenchMark/CLNDAT_EN.txt",
                             img_size = size,
                             augment = augment,
                             raw_img = True,
                             model = 'densenet',
                             zca = False)

    if mode in ['learning']:
        init = True
    elif mode in ['update', 'prediction']:
        init = False
    else:
        init = False

    obj = Detecter(output_type = output_type,
                   epoch = epoch, batch = batch, log = log,
                   optimizer_type = 'Adam',
                   learning_rate = lr,
                   dynamic_learning_rate = dlr,
                   beta1 = 0.9, beta2 = 0.999,
                   regularization = rr,
                   regularization_type = rtype,
                   checkpoint = checkpoint,
                   init = init,
                   size = size,
                   l1_norm = l1_norm)
    obj.construct()

    testdata = dataset.test.get_all_files()
    findings = [testdata[4][0]]
    root, ext = os.path.splitext(filename)
    img = dataset.test.img_process(filename, ext, augment = False)
    print(img.shape)
    ts = [img]
    x, y = obj.prediction(data = ts, roi = roi,
                          label_def = label_def, save_dir = './Pic',
                          filenames = [filename],
                          findings = findings,
                          roi_force = True)
    print("File name:", filename)
    print(y[0])



def get_results(outfile, testdata, batch, obj, roi, label_def,
                img_reader):
    with open(outfile, "w") as f:
        writer = csv.writer(f)
        ts, nums, filenames = [], [], []
        for i, t in enumerate(testdata[0]):
            ts.append(img_reader(t, augment = False)[0])
            filenames.append(t)
            nums.append(i)
            if len(ts) == batch:
                findings = [testdata[4][num] for num in nums]
                x, y = obj.prediction(data = ts, roi = roi,
                                      label_def = label_def, save_dir = './Pic',
                                      filenames = filenames,
                                      findings = findings)
                for j, num in enumerate(nums):
                    print(i, j, num)
                    print("File name:", testdata[3][num])
                    print(testdata[2][num])
                    print(y[j])
                    record = [x[j][0], x[j][1], testdata[1][num][0], testdata[1][num][1],
                              y[j][0], y[j][1],
                              y[j][2], y[j][3],
                              y[j][4], y[j][5],
                              y[j][6], y[j][7],
                              y[j][8], y[j][9],
                              y[j][10], y[j][11],
                              y[j][12], y[j][13],
                              testdata[2][num][0], testdata[2][num][1],
                              testdata[2][num][2], testdata[2][num][3],
                              testdata[2][num][4], testdata[2][num][5],
                              testdata[2][num][6], testdata[2][num][7],
                              testdata[2][num][8], testdata[2][num][9],
                              testdata[2][num][10], testdata[2][num][11],
                              testdata[2][num][12], testdata[2][num][13]
                              ]
                    writer.writerow(record)
                ts, nums, filenames = [], [], []


if __name__ == '__main__':
    main()
