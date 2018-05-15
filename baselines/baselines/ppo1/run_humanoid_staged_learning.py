#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import joblib
import tensorflow as tf
import numpy as np
from mpi4py import MPI
import os, errno

def callback(localv, globalv):
    if localv['iters_so_far'] % 10 != 0:
        return
    save_dict = {}
    variables = localv['pi'].get_variables()
    for i in range(len(variables)):
        cur_val = variables[i].eval()
        save_dict[variables[i].name] = cur_val

    save_dir = logger.get_dir() + '/' + (str(localv['env'].env.env.assist_schedule).replace(' ', ''))
    try:
        os.makedirs(save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    joblib.dump(save_dict, save_dir+'/policy_params_'+ str(localv['iters_so_far'])+'.pkl', compress=True)
    joblib.dump(save_dict, logger.get_dir() + '/policy_params' + '.pkl', compress=True)

def train_mirror(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_mirror_policy, pposgd_mirror
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    env.env.assist_timeout = 100.0

    def policy_fn(name, ob_space, ac_space):
        return mlp_mirror_policy.MlpMirrorPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                 hid_size=64, num_hid_layers=3, gmm_comp=1,
                                                 mirror_loss=True,
                                                 observation_permutation=np.array(
                                                     [0.0001, -1, 2, -3, -4, -11, 12, -13, 14, 15, 16, -5, 6, -7, 8, 9,
                                                      10, -17, 18, -19, -24, 25, -26, 27, -20, 21, -22, 23, \
                                                      28, 29, -30, 31, -32, -33, -40, 41, -42, 43, 44, 45, -34, 35, -36,
                                                      37, 38, 39, -46, 47, -48, -53, 54, -55, 56, -49, 50, -51, 52, 58,
                                                      57, 59]),
                                                 action_permutation=np.array(
                                                     [-6, 7, -8, 9, 10, 11, -0.001, 1, -2, 3, 4, 5, -12, 13, -14, -19,
                                                      20, -21, 22, -15, 16, -17, 18]))
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
    env.seed(seed+MPI.COMM_WORLD.Get_rank())
    gym.logger.setLevel(logging.WARN)

    previous_params = None
    iter_num = 0
    last_iter = False

    # if initialize from previous runs
    #previous_params = joblib.load('')
    #env.env.env.assist_schedule = []

    joblib.dump(str(env.env.env.__dict__), logger.get_dir() + '/env_specs.pkl', compress=True)

    reward_threshold = None
    while True:
        if not last_iter:
            rollout_length_thershold = env.env.env.assist_schedule[2][0] / env.env.env.dt
        else:
            rollout_length_thershold = None
        opt_pi, rew = pposgd_mirror.learn(env, policy_fn,
                max_timesteps=num_timesteps,
                timesteps_per_batch=int(2500),
                clip_param=0.2, entcoeff=0.0,
                optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                gamma=0.99, lam=0.95, schedule='linear',
                callback=callback,
                sym_loss_weight=4.0,
                positive_rew_enforce=False,
                init_policy_params = previous_params,
                reward_drop_bound=500,
                rollout_length_thershold = rollout_length_thershold,
                policy_scope='pi' + str(iter_num),
                return_threshold = reward_threshold,
            )
        if iter_num == 0:
            reward_threshold = 0.7 * rew
        if last_iter:
            reward_threshold = None
        iter_num += 1

        opt_variable = opt_pi.get_variables()
        previous_params = {}
        for i in range(len(opt_variable)):
            cur_val = opt_variable[i].eval()
            previous_params[opt_variable[i].name] = cur_val
        # update the assist schedule
        for s in range(len(env.env.env.assist_schedule)-1):
            env.env.env.assist_schedule[s][1] = np.copy(env.env.env.assist_schedule[s+1][1])
        env.env.env.assist_schedule[-1][1][0] *= 0.75
        env.env.env.assist_schedule[-1][1][1] *= 0.75
        if env.env.env.assist_schedule[-1][1][0] < 5.0:
            env.env.env.assist_schedule[-1][1][0] = 0.0
        if env.env.env.assist_schedule[-1][1][1] < 5.0:
            env.env.env.assist_schedule[-1][1][1] = 0.0
        zero_assist = True
        for s in range(len(env.env.env.assist_schedule)-1):
            for v in env.env.env.assist_schedule[s][1]:
                if v != 0.0:
                    zero_assist = False
        print('Current Schedule: ', env.env.env.assist_schedule)
        if zero_assist:
            last_iter = True
            print('Entering Last Iteration!')

    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartHumanWalker-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    logger.reset()
    logger.configure('data/ppo_'+args.env+str(args.seed)+'_energy03_vel15_1s_mirror4_velrew3_ab6_norotpen_dofpen11508_rew05xinit_thigh160_50springankle_1p2term_stagedcurriculum_075reduce_07rewthres_2kassist')
    train_mirror(args.env, num_timesteps=int(5000*4*800), seed=args.seed)


if __name__ == '__main__':
    main()
