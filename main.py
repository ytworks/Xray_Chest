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
    parser.add_argument('-log')
    parser.add_argument('-outfile')
    parser.add_argument('-mode', required = True)
    args = parser.parse_args()
    size = args.size if args.size != None else 256
    augment = True if args.augment != None else False
    checkpoint = args.checkpoint if args.checkpoint != None else './Model/Core.ckpt'
    lr = args.lr if args.lr != None else 0.0001
    dlr = args.dlr if args.dlr != None else 0.0
    rtype = args.rtype if args.rtype != None else 'L2'
    rr = args.rr if args.rr != None else 0.0
    epoch = args.epoch if args.epoch != None else 2
    batch = args.batch if args.batch != None else 5
    log = args.log if args.log != None else 2
    outfile = args.outfile if args.outfile != None else './Result/result.csv'
    if args.mode in ['learning']:
        init = True
    elif args.mode in ['update', 'prediction']:
        init = False
    else:
        init = False

    print("read dataset")
    dataset, label_def = read_data_sets(nih_datapath = ["./Data/Open/images/*.png"],
                             nih_supervised_datapath = "./Data/Open/Data_Entry_2017.csv",
                             nih_boxlist = "./Data/Open/BBox_List_2017.csv",
                             benchmark_datapath = ["./Data/CR_DATA/BenchMark/*/*.dcm"],
                             benchmark_supervised_datapath = "./Data/CR_DATA/BenchMark/CLNDAT_EN.txt",
                             kfold = 1,
                             img_size = size,
                             augment = augment,
                             zca = True)
    print("label definitions:")
    print(label_def)

    obj = Detecter(output_type = 'classified-softmax',
                   epoch = epoch, batch = batch, log = log,
                   optimizer_type = 'Adam',
                   learning_rate = lr,
                   dynamic_learning_rate = dlr,
                   beta1 = 0.9, beta2 = 0.999,
                   regularization = rr,
                   regularization_type = rtype,
                   checkpoint = checkpoint,
                   init = init,
                   size = size)
    obj.construct()
    if args.mode != 'prediction':
        logger.debug("Start learning")
        obj.learning(data = dataset,
                     validation_batch_num = 1)
        logger.debug("Finish learning")
    else:
        logger.debug("Skipped learning")
    testdata = dataset.test.get_all_data()
    with open(outfile, "w") as f:
        writer = csv.writer(f)
        for i, t in tqdm(enumerate(testdata[0])):
            x, _ = obj.prediction(data = [t], roi = True, label_def = label_def, save_dir = './Pic',
                                  filename = os.path.splitext(testdata[3][i]),
                                  path = testdata[4][i])
            print("File name:", testdata[3][i])
            print(x, testdata[1][i])
            writer.writerow([x[0][0], x[0][1], testdata[1][i][0], testdata[1][i][1]])



if __name__ == '__main__':
    main()
