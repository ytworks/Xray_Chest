cd `dirname $0`
cd ../

config="./experiments/adamw_4.ini"
CUDA_VISIBLE_DEVICES=0 python main.py -config $config > 20181109_adamw_4.out &

config="./experiments/adamw_5.ini"
CUDA_VISIBLE_DEVICES=1 python main.py -config $config > 20181109_adamw_4.out &

config="./experiments/adamw_6.ini"
CUDA_VISIBLE_DEVICES=2 python main.py -config $config > 20181109_adamw_4.out &

config="./experiments/adamw_7.ini"
CUDA_VISIBLE_DEVICES=3 python main.py -config $config > 20181109_adamw_4.out &
