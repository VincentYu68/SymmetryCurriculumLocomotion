from baselines.siggraph_script.training_utils import *
import gym
from baselines import logger
import numpy as np

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartHexapod-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    logger.reset()
    logger.configure('data/ppo_' + args.env + str(
        args.seed) + '_walk')

    env = gym.make(args.env)
    env.env.assist_timeout = 100.0
    env.env.target_vel = 2.0
    env.env.init_tv = 0.0
    env.env.final_tv = 2.0
    env.env.tv_endtime = 1.0
    env.env.energy_weight = 0.2
    env.env.alive_bonus = 4.0
    train_mirror_sig(env, num_timesteps=int(5000000), seed=args.seed, obs_perm=np.array(
                                                     [0.0001, -1, 2, -3, -4, 8, 9, 10, 5, 6, 7, 14, 15, 16, 11, 12, 13,
                                                      20, 21, 22, 17, 18, 19,
                                                      23, 24, -25, 26, -27, -28, 32, 33, 34, 29, 30, 31, 38, 39, 40, 35,
                                                      36, 37, 44, 45, 46, 41, 42, 43,
                                                      48, 47, 50, 49, 52, 51, 53]), act_perm=np.array(
                                                     [3, 4, 5, 0.0001, 1, 2, 9, 10, 11, 6, 7, 8, 15, 16, 17, 12, 13,
                                                      14]))

