#! /usr/bin/bash
cd `dirname $0`
cd ../../../
source activate tensorflow_p27
size='224'
augment='False'
checkpoint='./Model/run.ckpt'
outfile='./Result/result.csv'
epoch='500'
batch='32'
log='100'
lr='0.0001'
rr='0.1'
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

python main_pretraining.py  -mode learning -size $size \
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
