#! /usr/bin/bash
cd `dirname $0`
cd ../../../
source activate tensorflow_p27
size='224'
augment='True'
checkpoint='./Model/run.ckpt'
outfile='./Result/result.csv'
epoch='40000'
batch='32'
log='10000'
lr='0.0001'
rr='2'
l1_norm='1'
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
              -lr $lr

if [[ $FLAG == "debug" ]]; then
  echo "Not Stop"
else
  bash /home/ec2-user/git/mgmt/stop.sh
fi
