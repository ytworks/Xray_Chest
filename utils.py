#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv

def show_config(ini):
    '''
    設定ファイルの全ての内容を表示する（コメントを除く）
    '''
    for section in ini.sections():
        print '[%s]' % (section)
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
    print '%s.%s =%s' % (section, key, ini.get(section, key))

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
                x, y = obj.prediction(data=ts, roi=roi,
                                      label_def=label_def, save_dir='./Pic',
                                      filenames=filenames,
                                      suffixs=findings)
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
