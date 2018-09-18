#! /usr/bin/bash
cd `dirname $0`
cd ../

config="./settings/dev.ini"
python main.py -config $config
