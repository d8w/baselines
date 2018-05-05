one_million=1000000
ten_million=10000000
hundred_million=100000000
num_timesteps=$ten_million

# write log to files and no stdout
log_format=log,csv,json

env=PongNoFrameskip-v4
policy=lstm

log_dir=tests/$env-$policy

model_dir=models/$env-$policy

mode=test

time OPENAI_LOG_FORMAT=$log_format PYTHONPATH=. python baselines/a2c/run_atari.py --env $env --policy $policy --num-timesteps $num_timesteps --logdir=$log_dir --modeldir=$model_dir --mode=$mode

