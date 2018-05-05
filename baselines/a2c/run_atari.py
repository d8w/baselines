#!/usr/bin/env python3

from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.a2c.a2c import learn, build_model, Runner
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
from baselines.deepq.utils import save_state, load_state
from tqdm import tqdm
import time

def mk_env(env_id, num_timesteps, seed, policy, lrschedule, num_env, nsteps):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4)
    return policy_fn, env

def train(env_id, num_timesteps, seed, policy, lrschedule, num_env, nsteps, checkpoint_dir=None):
    policy_fn, env = mk_env(env_id, num_timesteps, seed, policy, lrschedule, num_env, nsteps)
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule,
            nsteps=nsteps, checkpoint_dir=checkpoint_dir)
    env.close()

def mk_model(env_id, num_timesteps, seed, policy, lrschedule, num_env, nsteps):
    """
    Create the A2C model
    """
    policy_fn, env = mk_env(env_id, num_timesteps, seed, policy, lrschedule, num_env, nsteps)
    nenvs, ob_space, ac_space, model = build_model(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule,
            nsteps=nsteps)
    return env, nenvs, model

def test(env, nenvs, model, nsteps=5, gamma=0.99, total_timesteps=100000, log_interval=100):
    """
    31 episodes ~ 1,00,000 steps
    """
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)

    nbatch = nenvs*nsteps
    tstart = time.time()
    for update in tqdm(range(1, total_timesteps//nbatch+1)):
        obs, states, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.dump_tabular()
    env.close()

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--workers', help='Number of workers', type=int, default=4)
    parser.add_argument('--nsteps', help='Number of LSTM steps', type=int, default=5)
    parser.add_argument('--logdir', help='Output logs to the folder', type=str, default=None)
    parser.add_argument('--modeldir', help='Save trained model to the dir or load a model from the dir for testing', type=str, default=None)
    parser.add_argument('--mode', help='', choices=['train', 'test'], default='train')
    args = parser.parse_args()
    logger.configure(dir=args.logdir)
    print(">>> Write log to {}".format(logger.get_dir()))
    if args.mode == "train":
        print(">>> Start training ...")
        try:
            train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
                policy=args.policy, lrschedule=args.lrschedule, num_env=args.workers, nsteps=args.nsteps, checkpoint_dir=args.modeldir)
        except KeyboardInterrupt:
            if args.modeldir is not None:
                print(">>> Save model to {}".format(args.modeldir))
                save_state(args.modeldir)
    elif args.mode == 'test':
        print(">>> Start testing ...")
        if args.modeldir is not None:
            print(">>> Load model from {}".format(args.modeldir))
            # Load the model
            env, nenvs, model = mk_model(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
                policy=args.policy, lrschedule=args.lrschedule, num_env=args.workers, nsteps=args.nsteps)
            load_state(args.modeldir)
            # Test
            test(env, nenvs, model, nsteps=args.nsteps)
        else:
            print(">>> Error: model file is not specified")

if __name__ == '__main__':
    main()
