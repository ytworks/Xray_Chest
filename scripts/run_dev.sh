#! /usr/bin/bash
cd `dirname $0`
cd ../


FLAG=""
VM="1"
while getopts d:v OPT
do
  case $OPT in
    d) FLAG=$OPTARG;;
    v) VM="0";;
  esac
done
if [[ $VM == "1" ]]; then
  source activate tensorflow_p27
fi

config="./settings/dev.ini"
python main.py -config $config

if [[ $FLAG == "debug" ]]; then
  echo "Not Stop"
else
  bash /home/ec2-user/git/mgmt/stop.sh
fi
