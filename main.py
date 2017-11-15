#!/usr/bin/env python
# -*- coding: utf-8 -*-
from make_mini_batch import read_data_sets
from detecter_original_v1 import Detecter


def main():
    dataset = read_data_sets(nih_datapath = ["./Data/Open/images/*.png"],
                             nih_supervised_datapath = "./Data/Open/Data_Entry_2017.csv",
                             nih_boxlist = "./Data/Open/BBox_List_2017.csv",
                             benchmark_datapath = ["./Data/CR_DATA/BenchMark/*/*.dcm"],
                             benchmark_supervised_datapath = "./Data/CR_DATA/BenchMark/CLNDAT_EN.txt",
                             kfold = 1,
                             img_size = 512,
                             augment = True,
                             zca = True)

    obj = Detecter(output_type = 'classified-softmax',
                   epoch = 100, batch = 32, log = 10,
                   optimizer_type = 'Adam',
                   learning_rate = 0.0001,
                   dynamic_learning_rate = 0.0,
                   beta1 = 0.9, beta2 = 0.999,
                   regularization = 0.0,
                   regularization_type = 'L2',
                   checkpoint = './Storages/Core.ckpt',
                   init = True,
                   size = 256)
    obj.construct()

if __name__ == '__main__':
    main()
