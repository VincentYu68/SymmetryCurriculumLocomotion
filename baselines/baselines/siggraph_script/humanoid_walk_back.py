from baselines.siggraph_script.training_utils import *
import gym
from baselines import logger
import numpy as np

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartHumanWalker-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    logger.reset()
    logger.configure('data/ppo_' + args.env + str(
        args.seed) + '_walk_back')

    env = gym.make(args.env)
    env.env.assist_timeout = 100.0
    env.env.target_vel = -1.5
    env.env.init_tv = 0.0
    env.env.final_tv = -1.5
    env.env.tv_endtime = 0.5
    env.env.energy_weight = 0.3
    env.env.alive_bonus_rew = 6.0
    train_mirror_sig(env, num_timesteps=int(5000000), seed=args.seed, obs_perm=np.array(
                                                     [0.0001, -1, 2, -3, -4, -11, 12, -13, 14, 15, 16, -5, 6, -7, 8, 9,
                                                      10, -17, 18, -19, -24, 25, -26, 27, -20, 21, -22, 23, \
                                                      28, 29, -30, 31, -32, -33, -40, 41, -42, 43, 44, 45, -34, 35, -36,
                                                      37, 38, 39, -46, 47, -48, -53, 54, -55, 56, -49, 50, -51, 52, 58,
                                                      57, 59]), act_perm=np.array(
                                                     [-6, 7, -8, 9, 10, 11, -0.001, 1, -2, 3, 4, 5, -12, 13, -14, -19,
                                                      20, -21, 22, -15, 16, -17, 18]))

