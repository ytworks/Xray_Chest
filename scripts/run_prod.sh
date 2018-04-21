#! /usr/bin/bash
cd `dirname $0`
cd ../
source activate tensorflow_p27

FLAG=""
while getopts d: OPT
do
  case $OPT in
    d) FLAG=$OPTARG;;
  esac
done

config="./settings/prod.ini"
python main.py -config $config

if [[ $FLAG == "debug" ]]; then
  echo "Not Stop"
else
  bash /home/ubuntu/git/mgmt/stop.sh
fi
