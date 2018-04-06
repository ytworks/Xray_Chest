#! /usr/bin/bash
cd `dirname $0`
cd ../../../
#source activate tensorflow_p27
size='512'
augment='True'
checkpoint='./Model/run.ckpt'
epoch='40000'
batch='16'
lr='0.001'
rr='0.0'
l1_norm='0.0'
roi='True'
output_type='classified-softmax'
#output_type='classified-squared-hinge'

FLAG=""
while getopts d: OPT
do
  case $OPT in
    d) FLAG=$OPTARG;;
  esac
done

python main_prediction.py  -mode prediction -size $size \
              -augment $augment \
              -checkpoint $checkpoint \
              -epoch $epoch \
              -batch $batch \
              -rr $rr \
              -l1_norm $l1_norm \
              -output_type $output_type \
              -lr $lr \
              -roi $roi \
              -file $1

if [[ $FLAG == "debug" ]]; then
  echo "Not Stop"
else
  bash /home/ec2-user/git/mgmt/stop.sh
fi
