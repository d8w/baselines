# Write log in Tensorboard format
# time OPENAI_LOG_FORMAT=tensorboard OPENAI_LOGDIR=pong_dqn PYTHONPATH=. python baselines/deepq/experiments/run_atari.py --prioritized 0

# Write log in CSV
time PYTHONPATH=. python baselines/deepq/experiments/run_atari_lstm.py --env=SpaceInvadersNoFrameskip-v4 --summary_dir=tb-dqn-sapce-invaders-500K-buffer
