ONE_MILLION=1000000
TEN_MILLION=10000000
HUNDRED_MILLION=100000000
num_timesteps=$ONE_MILLION

# write log to files and no stdout
log_format=log,csv,json

env=AsteroidsNoFrameskip-v4
policy=lstm

model_dir=models/$env-$policy/$env-$policy

mode=test

# probability for selecting sticky actions
for sigma in `seq 0.1 0.1 0.9`; do
    log_dir=tests/$env-$policy-sigma$sigma

    time OPENAI_LOG_FORMAT=$log_format PYTHONPATH=. python baselines/a2c/run_atari.py --env $env --policy $policy --num-timesteps $num_timesteps --logdir=$log_dir --modeldir=$model_dir --mode=$mode --sigma=$sigma
done

