#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import numpy as np
from six.moves import configparser as cp
import six
from sklearn.metrics import roc_curve, auc
import scipy
import cv2
import skimage


def image_process(img, channel):
    if channel == 3:
        img = np.stack((img, img, img), axis=-1)
    else:
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
    return img


def config_list(args):
    if six.PY2:
        config = cp.SafeConfigParser()
    else:
        config = cp.ConfigParser()
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
    epoch = config.getfloat('DLParams', 'epoch')
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
    optimizer_type = config.get('DLParams', 'optimizer_type')
    return size, augment, checkpoint, lr, dlr, rtype, rr, l1_norm, dumping_rate, dumping_period, epoch, batch, log, tflog, ds, roi, output_type, outfile, mode, step, split_mode, network_mode, auc_file, validation_set, optimizer_type, config


def show_config(ini):
    '''
    設定ファイルの全ての内容を表示する（コメントを除く）
    '''
    for section in ini.sections():
        print('[%s]' % (section))
        show_section(ini, section)


def show_section(ini, section):
    '''
    設定ファイルの特定のセクションの内容を表示する
    '''
    for key in ini.options(section):
        show_key(ini, section, key)


def show_key(ini, section, key):
    '''
    設定ファイルの特定セクションの特定のキー項目（プロパティ）の内容を表示する
    '''
    print('%s.%s =%s' % (section, key, ini.get(section, key)))


def get_roc_curve(filename, diags):
    f = csv.reader(open(filename, 'r'), lineterminator='\n')
    test, prob = [], []
    test_diag, prob_diag = [[] for i in range(len(diags))], [
        [] for i in range(len(diags))]
    for row in f:
        test.append(int(float(row[2])))
        prob.append(float(row[0]))
        for i in range(len(diags)):
            test_diag[i].append(int(float(row[i + 15 + 4])))
            prob_diag[i].append(float(row[i + 4]))
    for i, n in enumerate(diags):
        fpr, tpr, thresholds = roc_curve(
            test_diag[i], prob_diag[i], pos_label=1)
        roc = [[fpr[j], tpr[j], thresholds[j]] for j in range(len(fpr))]
        roc = np.array(roc)
        np.save('./Config/' + n + '.npy', roc)


def get_results(outfile, testdata, batch, obj, roi, label_def,
                img_reader):
    with open(outfile, "w") as f:
        writer = csv.writer(f)
        ts, nums, filenames = [], [], []
        for i, t in enumerate(testdata[0]):
            ts.append(img_reader(t, augment=False)[0])
            filenames.append(t)
            nums.append(i)
            if len(ts) == batch or len(testdata[0]) == i + 1:
                findings = [testdata[4][num] for num in nums]
                x, y, _, _ = obj.prediction(data=ts, roi=roi,
                                            label_def=label_def, save_dir='./Pic',
                                            filenames=filenames)
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
                              y[j][12], y[j][13], y[j][14],
                              testdata[2][num][0], testdata[2][num][1],
                              testdata[2][num][2], testdata[2][num][3],
                              testdata[2][num][4], testdata[2][num][5],
                              testdata[2][num][6], testdata[2][num][7],
                              testdata[2][num][8], testdata[2][num][9],
                              testdata[2][num][10], testdata[2][num][11],
                              testdata[2][num][12], testdata[2][num][13], testdata[2][num][14]
                              ]
                    writer.writerow(record)
                ts, nums, filenames = [], [], []
