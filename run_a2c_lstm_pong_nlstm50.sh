ONE_MILLION=1000000
TEN_MILLION=10000000
HUNDRED_MILLION=100000000
num_timesteps=$TEN_MILLION

# write log to files and no stdout
log_format=log,csv,json

env=PongNoFrameskip-v4
policy=lstm
nlstm=50
log_dir=exp/$env-$policy-nlstm${nlstm}

model_dir=models/$env-$policy/$env-$policy-nlstm${nlstm}

time OPENAI_LOG_FORMAT=$log_format PYTHONPATH=. python baselines/a2c/run_atari.py --env $env --policy $policy --num-timesteps $num_timesteps --logdir=$log_dir --modeldir=$model_dir --nsteps $nlstm

