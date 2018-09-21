#! /usr/bin/bash
cd `dirname $0`
cd ../

while getopts i:o: OPT
do
  case $OPT in
    i) FILE=$OPTARG;;
    o) DIR=$OPTARG;;
  esac
done


config="./settings/prediction_dev.ini"
echo $FILE
python main_prediction.py -config $config -file $FILE -dir $DIR
