#! /usr/bin/env python
# coding:utf-8
import csv
from DICOMReader.DICOMReader import dicom_to_np
import cv2
import numpy as np

def main():
    csv_file = "./Data/CR_DATA/BenchMark/CLNDAT_EN.txt"
    f = open(csv_file, "rU")
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if row != []:
            print "./Data/CR_DATA/BenchMark/"+row[0].replace("IMG", "dcm")
            img, _ = dicom_to_np("./Data/CR_DATA/BenchMark/Nodule154images/"+row[0].replace("IMG", "dcm"))
            img = img.astype(np.float32) / 4095 * 255
            img = img.astype(np.int32)
            img = np.stack((img, img, img), axis=2)
            cv2.imwrite('./Pic/'+row[0].replace("IMG", "png"), img)
            img_a = cv2.rectangle(img, (int(row[5])-100, int(row[6]) - 100), (int(row[5])+100, int(row[6])+100), (0, 0, 255), 30)
            cv2.imwrite('./Pic/'+row[0].replace(".IMG", "_annotation.png"), img_a)

if __name__=='__main__':
    main()
