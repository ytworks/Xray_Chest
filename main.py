#!/usr/bin/env python
# -*- coding: utf-8 -*-
from make_mini_batch import read_data_sets
from detecter_original_v1 import Detecter
from logging import getLogger, StreamHandler
logger = getLogger(__name__)
sh = StreamHandler()
logger.addHandler(sh)
logger.setLevel(10)


def main():
    dataset = read_data_sets(nih_datapath = ["./Data/Open/images/*.png"],
                             nih_supervised_datapath = "./Data/Open/Data_Entry_2017.csv",
                             nih_boxlist = "./Data/Open/BBox_List_2017.csv",
                             benchmark_datapath = ["./Data/CR_DATA/BenchMark/*/*.dcm"],
                             benchmark_supervised_datapath = "./Data/CR_DATA/BenchMark/CLNDAT_EN.txt",
                             kfold = 1,
                             img_size = 256,
                             augment = True,
                             zca = True)

    obj = Detecter(output_type = 'classified-softmax',
                   epoch = 3, batch = 5, log = 1,
                   optimizer_type = 'Adam',
                   learning_rate = 0.0001,
                   dynamic_learning_rate = 0.0,
                   beta1 = 0.9, beta2 = 0.999,
                   regularization = 0.0,
                   regularization_type = 'L2',
                   checkpoint = './Model/Core.ckpt',
                   init = True,
                   size = 256)
    obj.construct()
    obj.learning(data = dataset,
                 validation_batch_num = 1)
    logger.debug("Finish learning")
    testdata = dataset.test.get_all_data()
    for i, t in enumerate(testdata[0]):
        x = obj.prediction(data = [t])
        logger.debug(x)


if __name__ == '__main__':
    main()
