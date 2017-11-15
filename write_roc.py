#! /usr/bin/env python
# coding:utf-8

import numpy as np
import matplotlib.pyplot as pl
from sklearn.metrics import roc_curve, auc
import csv
import sys

p = sys.argv

f = csv.reader(open(p[1], 'r'), lineterminator='\n')
test, prob = [], []
for row in f:
    test.append(int(float(row[2])))
    prob.append(float(row[0]))

fpr, tpr, thresholds = roc_curve(test, prob, pos_label = 1)
print(fpr)
roc_auc = auc(fpr, tpr)
print "Area under the ROC curve : %f" % roc_auc

pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
for i in range(0, len(fpr), int(len(fpr) / 5)):
    pl.text(fpr[i],tpr[i],'%0.5f'%thresholds[i],fontsize=8)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('ROC : Cross Validation of Benchmark')
pl.legend(loc="lower right")
pl.show()
