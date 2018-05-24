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
import tensorflow as tf
from utils import *
logger = getLogger(__name__)
sh = StreamHandler()
logger.addHandler(sh)
logger.setLevel(10)

'''
Todo: main_predictionとの重複部分の共通化
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config')
    args = parser.parse_args()

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
    dumping_rate = config.getfloat('DLParams', 'dumping_rate')
    dumping_period = config.getint('DLParams', 'dumping_period')
    epoch = config.getint('DLParams', 'epoch')
    batch = config.getint('DLParams', 'batch')
    log = config.getint('LogParams', 'log_period')
    tflog = config.getint('LogParams', 'tflog_period')
    ds = config.get('InputParams', 'dataset')
    roi = config.getboolean('Mode', 'roi_prediction')
    output_type = config.get('DLParams', 'output_type')
    outfile = config.get('OutputParams', 'outfile')
    mode = config.get('Mode', 'running_mode')
    step = config.getint('DLParams', 'step')
    split_mode = config.get('Mode', 'split_mode')
    network_mode = config.get('Mode', 'network_mode')
    auc_file = config.get('OutputParams', 'auc_file')
    validation_set = config.getboolean('Mode', 'validation_set')

    if mode in ['learning']:
        init = True
    elif mode in ['update', 'prediction']:
        init = False
    else:
        init = False

    print("read dataset")
    dataset, label_def = read_data_sets(nih_datapath=["./Data/Open/images/*.png"],
                                        nih_supervised_datapath="./Data/Open/Data_Entry_2017_v2.csv",
                                        nih_boxlist="./Data/Open/BBox_List_2017.csv",
                                        benchmark_datapath=[
                                            "./Data/CR_DATA/BenchMark/*/*.dcm"],
                                        benchmark_supervised_datapath="./Data/CR_DATA/BenchMark/CLNDAT_EN.txt",
                                        split_file_dir="./Data",
                                        split_mode=split_mode,
                                        img_size=size,
                                        augment=augment,
                                        raw_img=True,
                                        model='densenet',
                                        zca=False,
                                        validation_set=validation_set)
    print("label definitions:")
    print(label_def)

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
    if mode != 'prediction':
        logger.debug("Start learning")
        num = int(len(dataset.val.files) / batch) + 1
        obj.learning(data=dataset,
                     validation_batch_num=num)
        logger.debug("Finish learning")
    else:
        logger.debug("Skipped learning")
    confdata = dataset.conf.get_all_files()
    get_results(outfile.replace('result', 'conf_result'), confdata, batch, obj, roi, label_def,
                img_reader=dataset.conf.img_reader)
    testdata = dataset.test.get_all_files()
    get_results(outfile.replace('result', 'nih_result'), testdata, batch, obj, roi, label_def,
                img_reader=dataset.test.img_reader)
    # sensivity / specifity table
    get_roc_curve(filename=outfile.replace(
        'result', 'nih_result'), diags=label_def)


if __name__ == '__main__':
    main()
