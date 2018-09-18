#! /usr/bin/bash
cd `dirname $0`
cd ../

config="./settings/prod.ini"
python main.py -config $config
