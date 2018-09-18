#! /usr/bin/bash
cd `dirname $0`
cd ../

config="./settings/prod_roi.ini"
python main.py -config $config
