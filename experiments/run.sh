cd `dirname $0`
cd ../

config="./experiments/adamw_5_focal.ini"
 CUDA_VISIBLE_DEVICES=0 nohup python main.py -config $config > 20181109_adamw_5_focal.out &

config="./experiments/adamw_5.ini"
CUDA_VISIBLE_DEVICES=1 nohup python main.py -config $config > 20181109_adamw_5.out &

config="./experiments/adamw_8.ini"
CUDA_VISIBLE_DEVICES=2 nohup python main.py -config $config > 20181109_adamw_8.out &

config="./experiments/adamw_8_focal.ini"
CUDA_VISIBLE_DEVICES=3 nohup python main.py -config $config > 20181109_adamw_8_focal.out &
