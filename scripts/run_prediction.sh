#! /usr/bin/bash
cd `dirname $0`
cd ../

FLAG=""
VM="1"
while getopts i:d:vo: OPT
do
  case $OPT in
    d) FLAG=$OPTARG;;
    v) VM="0";;
    i) FILE=$OPTARG;;
    o) DIR=$OPTARG;;
  esac
done
if [[ $VM == "1" ]]; then
  source activate tensorflow_p27
fi

config="./settings/prediction.ini"
echo $FLAG
echo $FILE
python main_prediction.py -config $config -file $FILE -dir $DIR

if [[ $FLAG == "debug" ]]; then
  echo "Not Stop"
else
  bash /home/ubuntu/git/mgmt/stop.sh
fi
