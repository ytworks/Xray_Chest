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
    def __init__(self):
        pass


def read_data_sets(mode,
                   nih_datapath,
                   nih_supervised_datapath,
                   nih_boxlist,
                   benchmark_datapath,
                   benchmark_supervised_datapath,
                   kfold = 3,
                   img_size = 512,
                   augment = True,
                   zca = True):
    class DataSets(object):
        pass
    data_sets = DataSets()
