#! /usr/bin/bash

size='256'
augment='True'
checkpoint='./Model/run.ckpt'
outfile='./Result/result.csv'
epoch='1'
batch='20'
log='500'
#python main.py -mode learning -size $size \
#              -augment $augment \
#              -checkpoint $checkpoint \
#              -epoch $epoch \
#              -batch $batch \
#              -log $log
bash /home/ec2-user/git/mgmt/stop.sh
