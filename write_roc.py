#! /usr/bin/env python
# coding:utf-8

import numpy as np
import matplotlib.pyplot as pl
from sklearn.metrics import roc_curve, auc

import csv
import sys


def write_fig(test, prob, figname):
    fpr, tpr, thresholds = roc_curve(test, prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)
    pl.figure()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
    for i in range(0, len(fpr), int(len(fpr) / 5)):
        pl.text(fpr[i], tpr[i], '%0.5f' % thresholds[i], fontsize=8)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('ROC')
    pl.legend(loc="lower right")
    pl.savefig("./Result/" + figname)
    return roc_auc


p = sys.argv

diags = ['Non-Fibrosis', 'Fibrosis']

f = csv.reader(open(p[1], 'r'), lineterminator='\n')
test, prob = [], []
test_diag, prob_diag = [[] for i in range(len(diags))], [
    [] for i in range(len(diags))]
for row in f:
    test.append(int(float(row[2])))
    prob.append(float(row[0]))
    for i in range(len(diags)):
        test_diag[i].append(int(float(row[i + 2])))
        prob_diag[i].append(float(row[i]))
rocs = []
for i, n in enumerate(diags):
    try:
        print(n)
        roc = write_fig(test_diag[i], prob_diag[i], n + ".png")
        if not n == 'No Findings':
            rocs.append(roc)
    except:
        print(n, "error")
print("average:", np.mean(rocs))

#write_fig(test, prob, "judgement.png")
