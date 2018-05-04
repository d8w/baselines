ONE_MILLION=1000000
TEN_MILLION=10000000
HUNDRED_MILLION=100000000
num_timesteps=$TEN_MILLION

# write log to files and no stdout
log_format=log,csv,json,tensorboard

env=FrostbiteNoFrameskip-v4
policy=lstm

log_dir=exp/$env-$policy

time OPENAI_LOG_FORMAT=$log_format PYTHONPATH=. python baselines/a2c/run_atari.py --env $env --policy $policy --num-timesteps $num_timesteps --logdir=$log_dir
