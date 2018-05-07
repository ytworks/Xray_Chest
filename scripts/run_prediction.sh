#! /usr/bin/bash
cd `dirname $0`
cd ../
#source activate tensorflow_p27

FLAG=""
while getopts d:f: OPT
do
  case $OPT in
    d) FLAG=$OPTARG;;
    f) FILE=$OPTARG;;
  esac
done

config="./settings/prediction.ini"
echo $FLAG
echo $FILE
python main_prediction.py -config $config -file $FILE

if [[ $FLAG == "debug" ]]; then
  echo "Not Stop"
else
  bash /home/ubuntu/git/mgmt/stop.sh
fi
