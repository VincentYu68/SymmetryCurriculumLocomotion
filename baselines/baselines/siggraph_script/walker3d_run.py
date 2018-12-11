from baselines.siggraph_script.training_utils import *
import gym
from baselines import logger
import numpy as np

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartWalker3d-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    logger.reset()
    logger.configure('data/ppo_' + args.env + str(
        args.seed) + '_run')

    env = gym.make(args.env)
    env.env.assist_timeout = 100.0
    env.env.target_vel = 5.0
    env.env.init_tv = 0.0
    env.env.final_tv = 5.0
    env.env.tv_endtime = 3.0
    env.env.energy_weight = 0.3
    env.env.alive_bonus = 7.0
    env.env.foot_lift_weight = 0.0
    train_mirror_sig(env, num_timesteps=int(5000 * 4 * 800), seed=args.seed, obs_perm=np.array(
                                                     [0.0001, -1, 2, -3, -4, -5, -6, 7, 14, -15, -16, 17, 18, -19, 8,
                                                      -9, -10, 11, 12, -13,
                                                      20, 21, -22, 23, -24, -25, -26, -27, 28, 35, -36, -37, 38, 39,
                                                      -40, 29, -30, -31, 32, 33,
                                                      -34, 42, 41, 43]), act_perm=np.array(
                                                     [-0.0001, -1, 2, 9, -10, -11, 12, 13, -14, 3, -4, -5, 6, 7, -8]))

