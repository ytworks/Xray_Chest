cd `dirname $0`
cd ../

config="./experiments/adamw_4.ini"
nohup CUDA_VISIBLE_DEVICES=0 python main.py -config $config > 20181109_adamw_5_focal.out &

config="./experiments/adamw_5.ini"
nohup CUDA_VISIBLE_DEVICES=1 python main.py -config $config > 20181109_adamw_5.out &

config="./experiments/adamw_6.ini"
nohup CUDA_VISIBLE_DEVICES=2 python main.py -config $config > 20181109_adamw_8.out &

config="./experiments/adamw_7.ini"
nohup CUDA_VISIBLE_DEVICES=3 python main.py -config $config > 20181109_adamw_8_focal.out &
