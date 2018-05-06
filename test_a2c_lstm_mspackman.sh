ONE_MILLION=1000000
TEN_MILLION=10000000
HUNDRED_MILLION=100000000
num_timesteps=$ONE_MILLION

# write log to files and no stdout
log_format=log,csv,json

env=MsPacmanNoFrameskip-v4
policy=lstm

log_dir=tests/$env-$policy

model_dir=models/$env-$policy/$env-$policy

mode=test
sigma=0.5 # probability for selecting sticky actions

time OPENAI_LOG_FORMAT=$log_format PYTHONPATH=. python baselines/a2c/run_atari.py --env $env --policy $policy --num-timesteps $num_timesteps --logdir=$log_dir --modeldir=$model_dir --mode=$mode --sigma=$sigma

