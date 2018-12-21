__author__ = 'yuwenhao'

import gym
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import sys, os, time, errno

import joblib
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from gym import wrappers
import tensorflow as tf
from baselines.ppo1 import mlp_policy, pposgd_simple
import baselines.common.tf_util as U
import pydart2.utils.transformations as trans
import json

np.random.seed(1)

def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                hid_size=64, num_hid_layers=3, gmm_comp=1)

def save_one_frame_shape(env, fpath, step):
    robo_skel = env.env.robot_skeleton
    data = []
    for b in robo_skel.bodynodes:
        if len(b.shapenodes) == 0:
            continue
        if 'cover' in b.name:
            continue
        shape_transform = b.T.dot(b.shapenodes[0].relative_transform()).tolist()
        #pos = trans.translation_from_matrix(shape_transform)
        #rot = trans.euler_from_matrix(shape_transform)
        shape_class = str(type(b.shapenodes[0].shape))
        if 'Mesh' in shape_class:
            stype = 'Mesh'
            path = b.shapenodes[0].shape.path()
            scale = b.shapenodes[0].shape.scale().tolist()
            sub_data = [path, scale]
        elif 'Box' in shape_class:
            stype = 'Box'
            sub_data = b.shapenodes[0].shape.size().tolist()
        elif 'Ellipsoid' in shape_class:
            stype = 'Ellipsoid'
            sub_data = b.shapenodes[0].shape.size().tolist()
        elif 'MultiSphere' in shape_class:
            stype = 'MultiSphere'
            sub_data = b.shapenodes[0].shape.spheres()
            for s in range(len(sub_data)):
                sub_data[s]['pos'] = sub_data[s]['pos'].tolist()

        data.append([stype, b.name, shape_transform, sub_data])
    file = fpath + '/frame_' + str(step)+'.txt'
    json.dump(data, open(file, 'w'))


if __name__ == '__main__':
    save_render_data = False
    interpolate = 0
    prev_state = None
    render_step = 0
    render_path = 'render_data/' + 'humanoid_walk'
    try:
        os.makedirs(render_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if len(sys.argv) > 1:
        env = gym.make(sys.argv[1])
    else:
        env = gym.make('DartWalker3d-v1')

    if hasattr(env.env, 'disableViewer'):
        env.env.disableViewer = False

    # manually set the target velocities for different tasks
    if len(sys.argv) > 2:
        policy_directory = '/'.join(sys.argv[2].split('/')[0:-1])+'/' # put data back to the folder that stores the policies
        if sys.argv[2][0] == '/':
            policy_directory = '/' + policy_directory

        save_directory = policy_directory + '/stats'

        try:
            os.makedirs(save_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        if sys.argv[1] == 'DartWalker3d-v1':
            env.env.assist_timeout = 0.0
            if 'walk' in sys.argv[2]:    # walk task
                env.env.final_tv = 1.0
                env.env.tv_endtime = 0.5
            if 'run' in sys.argv[2]:    # run task
                env.env.final_tv = 5.0
                env.env.tv_endtime = 2.0

        if sys.argv[1] == 'DartHumanWalker-v1':
            env.env.assist_timeout = 0
            if 'walk' in sys.argv[2]:    # walk task
                env.env.final_tv = 1.5
                env.env.tv_endtime = 0.5
            if 'walk_back' in sys.argv[2]:    # walk back task
                env.env.final_tv = -1.5
                env.env.tv_endtime = 0.5
            if 'run' in sys.argv[2]:    # run task
                env.env.final_tv = 5.0
                env.env.tv_endtime = 3.0

        if sys.argv[1] == 'DartDogRobot-v1':
            env.env.assist_timeout = 0.0
            if 'walk' in sys.argv[2]:    # walk task
                env.env.final_tv = 2.0
                env.env.tv_endtime = 1.0
            if 'run' in sys.argv[2]:    # run task
                env.env.final_tv = 7.0
                env.env.tv_endtime = 3.0

        if sys.argv[1] == 'DartHexapod-v1':
            env.env.assist_timeout = 0.0
            if 'walk' in sys.argv[2]:    # walk task
                env.env.final_tv = 2.0
                env.env.tv_endtime = 1.0
            if 'run' in sys.argv[2]:    # run task
                env.env.final_tv = 4.0
                env.env.tv_endtime = 2.0



    record = False
    if len(sys.argv) > 3:
        record = int(sys.argv[3]) == 1
    if record:
        env_wrapper = wrappers.Monitor(env, save_directory, force=True)
    else:
        env_wrapper = env

    if len(sys.argv) > 4:
        env.env.visualize = int(sys.argv[4]) == 1
    if hasattr(env.env, 'reset_range'):
        env.env.reset_range = 0.0

    sess = tf.InteractiveSession()

    policy = None
    if len(sys.argv) > 2:
        policy_params = joblib.load(sys.argv[2])
        ob_space = env.observation_space
        ac_space = env.action_space
        policy = policy_fn("pi", ob_space, ac_space)

        U.initialize()

        cur_scope = policy.get_variables()[0].name[0:policy.get_variables()[0].name.find('/')]
        orig_scope = list(policy_params.keys())[0][0:list(policy_params.keys())[0].find('/')]
        vars = policy.get_variables()

        for i in range(len(policy.get_variables())):
            assign_op = policy.get_variables()[i].assign(
                policy_params[policy.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
            sess.run(assign_op)

        if 'curriculum' in sys.argv[2] and 'policy_params.pkl' in sys.argv[2]:
            if os.path.isfile(sys.argv[2].replace('policy_params.pkl', 'init_poses.pkl')):
                init_qs, init_dqs = joblib.load(sys.argv[2].replace('policy_params.pkl', 'init_poses.pkl'))
                env.env.init_qs = init_qs
                env.env.init_dqs = init_dqs

        '''ref_policy_params = joblib.load('data/ppo_DartHumanWalker-v1210_energy015_vel65_6s_mirror_up01fwd01ltl15_spinepen1yaw001_thighyawpen005_initbentelbow_velrew3_avg_dcon1_asinput_damping2kneethigh_thigh150knee100_curriculum_1xjoint_shoulder90_dqpen00001/policy_params.pkl')
        ref_policy = policy_fn("ref_pi", ob_space, ac_space)

        cur_scope = ref_policy.get_variables()[0].name[0:ref_policy.get_variables()[0].name.find('/')]
        orig_scope = list(ref_policy_params.keys())[0][0:list(ref_policy_params.keys())[0].find('/')]
        vars = ref_policy.get_variables()

        for i in range(len(ref_policy.get_variables())):
            assign_op = ref_policy.get_variables()[i].assign(
                ref_policy_params[ref_policy.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
            sess.run(assign_op)

        env.env.ref_policy = ref_policy'''


        #init_q, init_dq = joblib.load('data/skel_data/init_states.pkl')
        #env.env.init_qs = init_q
        #env.env.init_dqs = init_dq

    print('===================')

    o = env_wrapper.reset()

    rew = 0

    actions = []

    traj = 1
    ct = 0
    vel_rew = []
    action_pen = []
    deviation_pen = []
    ref_rewards = []
    ref_feat_rew = []
    rew_seq = []
    com_z = []
    x_vel = []
    foot_contacts = []
    contact_force = []
    both_contact_forces = []
    avg_vels = []
    d=False
    step = 0
    total_steps = 0

    save_qs = []
    save_dqs = []
    save_init_state = False

    while ct < traj:
        if policy is not None:
            ac, vpred = policy.act(step<0, o)  # apply stochastic policy at the beginning
            act = ac
        else:
            act = env.action_space.sample()
        actions.append(act)

        '''if env_wrapper.env.env.t > 3.0 and env_wrapper.env.env.t < 6.0:
            env_wrapper.env.env.robot_skeleton.bodynode('head').add_ext_force(np.array([-200, 0, 0]))'''
        o, r, d, env_info = env_wrapper.step(act)

        if 'action_pen' in env_info:
            action_pen.append(env_info['action_pen'])
        if 'vel_rew' in env_info:
            vel_rew.append(env_info['vel_rew'])
        rew_seq.append(r)
        if 'deviation_pen' in env_info:
            deviation_pen.append(env_info['deviation_pen'])
        if 'contact_force' in env_info:
            contact_force.append(env_info['contact_force'])
        if 'contact_forces' in env_info:
            both_contact_forces.append(env_info['contact_forces'])
        if 'ref_reward' in env_info:
            ref_rewards.append(env_info['ref_reward'])
        if 'ref_feat_rew' in env_info:
            ref_feat_rew.append(env_info['ref_feat_rew'])
        if 'avg_vel' in env_info:
            avg_vels.append(env_info['avg_vel'])

        com_z.append(o[1])
        foot_contacts.append(o[-2:])

        rew += r

        if len(sys.argv) > 4:
            if  env.env.visualize:
                env_wrapper.render()
        else:
            env_wrapper.render()
        step += 1
        total_steps += 1

        #time.sleep(0.1)
        if len(o) > 25:
            x_vel.append(env.env.robot_skeleton.dq[0])


        #if np.abs(env.env.t - env.env.tv_endtime) < 0.01:
        #    save_qs.append(env.env.robot_skeleton.q)
            save_dqs.append(env.env.robot_skeleton.dq)

        if save_render_data:
            cur_state = env.env.state_vector()
            if prev_state is not None and interpolate > 0:
                for it in range(interpolate):
                    int_state = (it+1)*1.0/(interpolate+1) * prev_state + (1-(it+1)*1.0/(interpolate+1)) * cur_state
                    env.env.set_state_vector(int_state)
                    save_one_frame_shape(env, render_path, render_step)
                    render_step += 1
            env.env.set_state_vector(cur_state)
            save_one_frame_shape(env, render_path, render_step)
            render_step += 1
            prev_state = env.env.state_vector()

        if d:
            step = 0
            if 'contact_locations' in env_info:
                c_loc = env_info['contact_locations']
                for j in range(len(c_loc[0]) - 1):
                    c_loc[0][j] = c_loc[0][j+1] - c_loc[0][j]
                for j in range(len(c_loc[1]) - 1):
                    c_loc[1][j] = c_loc[1][j + 1] - c_loc[1][j]
                print(np.mean(c_loc[0][0:-1], axis=0))
                print(np.mean(c_loc[1][0:-1], axis=0))
            ct += 1
            print('reward: ', rew)
            o=env_wrapper.reset()
            #break
    print('avg rew ', rew / traj)
    print('avg energy penalty: ', np.sum(action_pen)/total_steps)
    print('total vel rew: ', np.sum(vel_rew)/traj)

    if len(sys.argv) > 2:
        np.savetxt(save_directory+'/average_action_magnitude.txt', [np.sum(action_pen)/total_steps])

        if 'Walker' in sys.argv[1]: # measure SI for biped
            l_contact_total = 0
            r_contact_total = 0
            for i in range(len(actions)):
                l_contact_total += np.linalg.norm(actions[i][[0,1,2,3,4,5]])
                r_contact_total += np.linalg.norm(actions[i][[6,7,8,9,10,11]])
            print('total forces: ', l_contact_total, r_contact_total)
            print('SI: ', 2*np.abs(l_contact_total-r_contact_total)/(l_contact_total+r_contact_total))

            np.savetxt(save_directory + '/symmetry_index.txt', [2*np.abs(l_contact_total-r_contact_total)/(l_contact_total+r_contact_total)])








