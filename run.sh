#! /usr/bin/bash

source activate tensorflow_p27
size='256'
augment='True'
checkpoint='./Model/run.ckpt'
outfile='./Result/result.csv'
epoch='50000'
batch='10'
log='1000'
lr='0.001'
python main.py -mode learning -size $size \
              -augment $augment \
              -checkpoint $checkpoint \
              -epoch $epoch \
              -batch $batch \
              -log $log \
              -lr $lr

FLAG=""
while getopts d: OPT
do
  case $OPT in
    d) FLAG=$OPTARG;;
  esac
done
if [[ $FLAG == "debug" ]]; then
  echo "Not Stop"
else
  bash /home/ec2-user/git/mgmt/stop.sh
fi
