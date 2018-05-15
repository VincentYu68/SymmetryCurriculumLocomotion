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

def callback(localv, globalv):
    if localv['iters_so_far'] % 10 != 0:
        return
    save_dict = {}
    variables = localv['pi'].get_variables()
    for i in range(len(variables)):
        cur_val = variables[i].eval()
        save_dict[variables[i].name] = cur_val
    joblib.dump(save_dict, logger.get_dir()+'/policy_params_'+str(localv['iters_so_far'])+'.pkl', compress=True)
    joblib.dump(save_dict, logger.get_dir() + '/policy_params' + '.pkl', compress=True)


def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=3, gmm_comp=1)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=int(5000),
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
                        callback=callback
        )
    env.close()

def train_mirror(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_mirror_policy, pposgd_mirror
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)

    joblib.dump(str(env.env.__dict__), logger.get_dir() + '/env_specs.pkl', compress=True)

    def policy_fn(name, ob_space, ac_space):
        return mlp_mirror_policy.MlpMirrorPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                 hid_size=64, num_hid_layers=3, gmm_comp=1,
                                                 mirror_loss=True,
                                                 observation_permutation=np.array([0.0001,-1,2,-3,-4, -11,12,-13,14,15,16, -5,6,-7,8,9,10, -17,18, -19, -24,25,-26,27, -20,21,-22,23,\
                                          28,29,-30,31,-32,-33, -40,41,-42,43,44,45, -34,35,-36,37,38,39, -46,47, -48, -53,54,-55,56, -49,50,-51,52, 58,57, 59]),
        action_permutation=np.array([-6,7,-8, 9, 10,11,  -0.001,1,-2, 3, 4,5, -12,13, -14, -19,20,-21,22, -15,16,-17,18]))
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed+MPI.COMM_WORLD.Get_rank())
    ref_policy_params = None

    '''init_q, init_dq = joblib.load('data/skel_data/init_states.pkl')
    env.env.env.init_qs = init_q
    env.env.env.init_dqs = init_dq'''

    if env.env.env.use_ref_policy:
        ref_policy_params = joblib.load('data/ppo_DartHumanWalker-v1210_energy015_vel65_6s_mirror_up01fwd01ltl15_spinepen1yaw001_thighyawpen005_initbentelbow_velrew3_avg_dcon1_asinput_damping2kneethigh_thigh150knee100_curriculum_1xjoint_shoulder90_dqpen00001/policy_params.pkl')
    gym.logger.setLevel(logging.WARN)
    pposgd_mirror.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=int(2500),
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
            callback=callback,
            sym_loss_weight=0.0,
            positive_rew_enforce=False,
            #init_policy_params = joblib.load('data/ppo_DartHumanWalker-v1333_energy03_vel2_2s_mirror4_velrew3_adaptalivebonus_rew1xinit_thigh160_100springankle_stagedcurriculum_05reduce_07rewthres/policy_params.pkl'),
            reward_drop_bound=True,
            ref_policy_params = ref_policy_params,
            #init_policy_params = joblib.load('data/ppo_DartHumanWalker-v1124_energy25_vel3_kd1000_mirror_up1fwd01ltl15_spinepen1yaw001_thighyawpen005_initbentelbow_runningavg3_dcontrolconstraint1_asinput_damping2_fromvel3_kd500/policy_params.pkl')
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartHumanWalker-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    logger.reset()
    logger.configure('data/ppo_'+args.env+str(args.seed)+'_energy03_vel15_1s_mirror0_up03fwd03ltl15_spinepen1yaw001_ab3_thighyawpen005_velrewavg3_2s_dcon1_damping2kneethigh_thigh160knee100waist150_shoulder100_velrew15xinit_baseline')
    #logger.configure('data/ppo_'+args.env+str(args.seed)+'_energy05_bal_vel4smooth_mirror_up1fwd01ltl1_spinepen1yaw001_thighyawpen005_initbentelbow_velrew3_dcontrolconstraint1_strongerarm_asinput_treadmill')
    train_mirror(args.env, num_timesteps=int(5000*4*2500), seed=args.seed)

if __name__ == '__main__':
    main()
