#!/usr/bin/env python
# -*- coding: utf-8 -*-
from make_mini_batch import read_data_sets
from detection import Detecter
from logging import getLogger, StreamHandler
from tqdm import tqdm
import os
import csv
import argparse
import tensorflow as tf
from utils import *
logger = getLogger(__name__)
sh = StreamHandler()
logger.addHandler(sh)
logger.setLevel(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config')
    args = parser.parse_args()
    size, augment, checkpoint, lr, dlr, rtype, rr, l1_norm, dumping_rate, dumping_period, epoch, batch, log, tflog, ds, roi, output_type, outfile, mode, step, split_mode, network_mode, auc_file, validation_set, optimizer_type, config = config_list(
        args)

    if mode in ['learning']:
        init = True
    elif mode in ['update', 'prediction']:
        init = False
    else:
        init = False

    print("read dataset")
    dataset, label_def = read_data_sets(nih_datapath=[config.get('Data', 'nih_datapath')],
                                        nih_supervised_datapath=config.get('Data', 'nih_supervised_datapath'),
                                        nih_boxlist=config.get('Data', 'nih_boxlist'),
                                        split_file_dir=config.get('Data', 'split_file_dir'),
                                        nih_train_list=config.get('Data', 'nih_train_list'),
                                        nih_test_list=config.get('Data', 'nih_test_list'),
                                        split_mode=split_mode,
                                        img_size=size,
                                        augment=augment,
                                        raw_img=True,
                                        model=config.get('DLParams', 'preprocessing_type'),
                                        zca=False,
                                        validation_set=validation_set)
    print("label definitions:")
    print(label_def)

    obj = Detecter(output_type=output_type,
                   epoch=epoch, batch=batch, log=log,
                   optimizer_type=optimizer_type,
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
    testdata = dataset.test.get_all_files()
    get_results(outfile.replace('result', 'nih_result'), testdata, batch, obj, roi, label_def,
                img_reader=dataset.test.img_reader)
    # sensivity / specifity table
    get_roc_curve(filename=outfile.replace(
        'result', 'nih_result'), diags=label_def)


if __name__ == '__main__':
    main()
