#! /usr/bin/bash

source activate tensorflow_p27
size='256'
augment='False'
checkpoint='./Model/run.ckpt'
outfile='./Result/result.csv'
epoch='40000'
batch='10'
log='10000'
lr='0.0001'
rr='2'
l1_norm='1'
output_type='classified-softmax'
#output_type='classified-squared-hinge'

FLAG=""
while getopts d:m: OPT
do
  case $OPT in
    d) FLAG=$OPTARG;;
    m) MODEL=$OPTARG;;
  esac
done


if [[ $MODEL == 'pretrain' ]]; then
  MAIN='main_pretraining.py'
else
  MAIN='main.py'
fi
python $MAIN  -mode learning -size $size \
              -augment $augment \
              -checkpoint $checkpoint \
              -epoch $epoch \
              -batch $batch \
              -log $log \
              -rr $rr \
              -l1_norm $l1_norm \
              -output_type $output_type \
              -lr $lr

if [[ $FLAG == "debug" ]]; then
  echo "Not Stop"
else
  bash /home/ec2-user/git/mgmt/stop.sh
fi
