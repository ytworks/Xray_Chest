#! /usr/bin/bash

source activate tensorflow_p27
size='256'
augment='True'
checkpoint='./Model/run.ckpt'
outfile='./Result/result.csv'
epoch='501'
batch='10'
log='100'
lr='0.0001'
rr='2'
l1_norm='1'
#output_type='classified-softmax'
output_type='classified-squared-hinge'
python main.py -mode learning -size $size \
              -augment $augment \
              -checkpoint $checkpoint \
              -epoch $epoch \
              -batch $batch \
              -log $log \
              -rr $rr \
              -l1_norm $l1_norm \
              -output_type $output_type \
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
