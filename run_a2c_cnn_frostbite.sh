ONE_MILLION=1000000
TEN_MILLION=10000000
HUNDRED_MILLION=100000000
num_timesteps=$TEN_MILLION

# write log to files and no stdout
log_format=log,csv,json

env=FrostbiteNoFrameskip-v4
policy=cnn

log_dir=exp/$env-$policy

model_dir=models/$env-$policy/$env-$policy

time OPENAI_LOG_FORMAT=$log_format PYTHONPATH=. python baselines/a2c/run_atari.py --env $env --policy $policy --num-timesteps $num_timesteps --logdir=$log_dir --modeldir=$model_dir
