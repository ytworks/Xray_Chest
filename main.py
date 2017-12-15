#!/usr/bin/env python
# -*- coding: utf-8 -*-
from make_mini_batch import read_data_sets
from detecter_original_v1 import Detecter
from logging import getLogger, StreamHandler
from tqdm import tqdm
import os
import csv
import argparse
logger = getLogger(__name__)
sh = StreamHandler()
logger.addHandler(sh)
logger.setLevel(10)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-size')
    parser.add_argument('-augment')
    parser.add_argument('-checkpoint')
    parser.add_argument('-lr')
    parser.add_argument('-dlr')
    parser.add_argument('-rtype')
    parser.add_argument('-rr')
    parser.add_argument('-epoch')
    parser.add_argument('-batch')
    parser.add_argument('-l1_norm')
    parser.add_argument('-log')
    parser.add_argument('-outfile')
    parser.add_argument('-output_type')
    parser.add_argument('-mode', required = True)
    args = parser.parse_args()
    size = int(args.size) if args.size != None else 256
    augment = True if args.augment == 'True' else False
    checkpoint = args.checkpoint if args.checkpoint != None else './Model/Core.ckpt'
    lr = float(args.lr) if args.lr != None else 0.0001
    dlr = float(args.dlr) if args.dlr != None else 0.0
    rtype = args.rtype if args.rtype != None else 'L2'
    rr = float(args.rr) if args.rr != None else 0.0
    l1_norm = float(args.l1_norm) if args.l1_norm != None else 0.0
    epoch = int(args.epoch) if args.epoch != None else 2
    batch = int(args.batch) if args.batch != None else 5
    log = int(args.log) if args.log != None else 2
    output_type = args.output_type if args.output_type != None else 'classified-softmax'
    outfile = args.outfile if args.outfile != None else './Result/result.csv'
    if args.mode in ['learning']:
        init = True
    elif args.mode in ['update', 'prediction']:
        init = False
    else:
        init = False

    print("read dataset")
    dataset, label_def = read_data_sets(nih_datapath = ["./Data/Open/images/*.png"],
                             nih_supervised_datapath = "./Data/Open/Data_Entry_2017_v2.csv",
                             nih_boxlist = "./Data/Open/BBox_List_2017.csv",
                             benchmark_datapath = ["./Data/CR_DATA/BenchMark/*/*.dcm"],
                             benchmark_supervised_datapath = "./Data/CR_DATA/BenchMark/CLNDAT_EN.txt",
                             kfold = 1,
                             img_size = size,
                             augment = augment,
                             zca = True)
    print("label definitions:")
    print(label_def)

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
    if args.mode != 'prediction':
        logger.debug("Start learning")
        obj.learning(data = dataset,
                     validation_batch_num = int(250 / batch) + 1)
        logger.debug("Finish learning")
    else:
        logger.debug("Skipped learning")
    testdata = dataset.test.get_all_data()
    with open(outfile, "w") as f:
        writer = csv.writer(f)
        for i, t in tqdm(enumerate(testdata[0])):
            x, y = obj.prediction(data = [t], roi = True, label_def = label_def, save_dir = './Pic',
                                  filename = os.path.splitext(testdata[3][i]),
                                  path = testdata[4][i])
            print("File name:", testdata[3][i])
            print(x, testdata[1][i])
            print(y)
            record = [x[0][0], x[0][1], testdata[1][i][0], testdata[1][i][1],
                      y[0][0], y[0][1],
                      y[0][2], y[0][3],
                      y[0][4], y[0][5],
                      y[0][6], y[0][7],
                      y[0][8], y[0][9],
                      y[0][10], y[0][11],
                      y[0][12], y[0][13],
                      testdata[2][i][0], testdata[2][i][1],
                      testdata[2][i][2], testdata[2][i][3],
                      testdata[2][i][4], testdata[2][i][5],
                      testdata[2][i][6], testdata[2][i][7],
                      testdata[2][i][8], testdata[2][i][9],
                      testdata[2][i][10], testdata[2][i][11],
                      testdata[2][i][12], testdata[2][i][13]
                      ]
            writer.writerow(record)



if __name__ == '__main__':
    main()
