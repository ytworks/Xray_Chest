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
import json
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

    size, augment, checkpoint, lr, dlr, rtype, rr, l1_norm, dumping_rate, dumping_period, epoch, batch, log, tflog, ds, roi, output_type, outfile, mode, step, split_mode, network_mode, auc_file, validation_set = config_list(
        args)


    dataset, _ = read_data_sets(nih_datapath=["./Data/Open/images/*.png"],
                                nih_supervised_datapath="./Data/Open/Data_Entry_2017_v2.csv",
                                nih_boxlist="./Data/Open/BBox_List_2017.csv",
                                benchmark_datapath=[
                                            "./Data/CR_DATA/BenchMark/*/*.dcm"],
                                benchmark_supervised_datapath="./Data/CR_DATA/BenchMark/CLNDAT_EN.txt",
                                img_size=size,
                                augment=augment,
                                raw_img=True,
                                model='densenet',
                                zca=False,
                                validation_set=validation_set)

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
    img = dataset.test.img_process(filename, ext, augment=False)
    ts = [img]
    x, y, z = obj.prediction(data=ts, roi=roi,
                             label_def=label_list['label_def'], save_dir='./Pic',
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
    print(s)
    print(y.shape, z.shape)


if __name__ == '__main__':
    main()
