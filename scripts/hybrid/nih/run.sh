#! /usr/bin/bash
cd `dirname $0`
cd ../../../
source activate tensorflow_p27
size='256'
augment='True'
checkpoint='./Model/run.ckpt'
outfile='./Result/result.csv'
epoch='10000'
batch='256'
log='10'
lr='0.001'
rr='0.0'
l1_norm='0.0'
output_type='classified-softmax'
#output_type='classified-squared-hinge'

FLAG=""
while getopts d: OPT
do
  case $OPT in
    d) FLAG=$OPTARG;;
  esac
done

python main_hybrid.py  -mode learning -size $size \
              -augment $augment \
              -checkpoint $checkpoint \
              -epoch $epoch \
              -batch $batch \
              -log $log \
              -rr $rr \
              -l1_norm $l1_norm \
              -output_type $output_type \
              -lr $lr \
              -dataset nih

if [[ $FLAG == "debug" ]]; then
  echo "Not Stop"
else
  bash /home/ec2-user/git/mgmt/stop.sh
fi
