#! /usr/bin/bash
cd `dirname $0`
cd ../

config="./settings/local.ini"
python main.py -config $config
