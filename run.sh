#! /usr/bin/bash

source activate tensorflow_p27
size='256'
augment='True'
checkpoint='./Model/run.ckpt'
outfile='./Result/result.csv'
epoch='30000'
batch='20'
log='10'
lr='0.001'
python main.py -mode learning -size $size \
              -augment $augment \
              -checkpoint $checkpoint \
              -epoch $epoch \
              -batch $batch \
              -log $log \
              -lr $lr

bash /home/ec2-user/git/mgmt/stop.sh
