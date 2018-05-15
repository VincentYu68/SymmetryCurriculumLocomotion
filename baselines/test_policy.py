__author__ = 'yuwenhao'

import gym
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import sys, os, time, errno

import joblib
import numpy as np

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
        '''if hasattr(env.env, 'resample_MP'):
        env.env.resample_MP = False'''

    record = False
    if len(sys.argv) > 3:
        record = int(sys.argv[3]) == 1
    if record:
        env_wrapper = wrappers.Monitor(env, 'data/videos/', force=True)
    else:
        env_wrapper = env

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

    save_qs = []
    save_dqs = []
    save_init_state = False

    while ct < traj:
        if policy is not None:
            ac, vpred = policy.act(step<0, o)
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

        env_wrapper.render()
        step += 1

        #time.sleep(0.1)
        if len(o) > 25:
            x_vel.append(env.env.robot_skeleton.dq[0])

        if len(foot_contacts) > 400:
            if np.random.random() < 0.03:
                print('q ', np.array2string(env.env.robot_skeleton.q, separator=','))
                print('dq ', np.array2string(env.env.robot_skeleton.dq, separator=','))

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
    print('total energy penalty: ', np.sum(action_pen)/traj)
    print('total vel rew: ', np.sum(vel_rew)/traj)

    if 'Walker' in sys.argv[1]: # measure SI for biped
        l_contact_total = 0
        r_contact_total = 0
        for i in range(len(actions)):
            l_contact_total += np.linalg.norm(actions[i][[0,1,2,3,4,5]])
            r_contact_total += np.linalg.norm(actions[i][[6,7,8,9,10,11]])
        print('total forces: ', l_contact_total, r_contact_total)
        print('SI: ', 2*(l_contact_total-r_contact_total)/(l_contact_total+r_contact_total))

    if len(save_qs) > 0 and save_init_state:
        joblib.dump([save_qs, save_dqs], 'data/skel_data/init_states.pkl')

    if sys.argv[1] == 'DartWalker3d-v1' or sys.argv[1] == 'DartWalker3dSPD-v1':
        rendergroup = [[0,1,2], [3,4,5, 9,10,11], [6,12], [7,8, 12,13]]
        for rg in rendergroup:
            plt.figure()
            for i in rg:
                plt.plot(np.array(actions)[:, i])
    if sys.argv[1] == 'DartHumanWalker-v1':
        rendergroup = [[0,1,2, 6,7,8], [3,9], [4,5,10,11], [12,13,14], [15,16,7,18]]
        titles = ['thigh', 'knee', 'foot', 'waist', 'arm']
        for i,rg in enumerate(rendergroup):
            plt.figure()
            plt.title(titles[i])
            for i in rg:
                plt.plot(np.array(actions)[:, i])
    if sys.argv[1] == 'DartDogRobot-v1':
        rendergroup = [[0,1,2], [3, 4,5], [6,7,8],[9,10,11]]
        titles = ['rear right leg', 'rear left leg', 'front right leg', 'front left leg']
        for i,rg in enumerate(rendergroup):
            plt.figure()
            plt.title(titles[i])
            for i in rg:
                plt.plot(np.array(actions)[:, i])
    if sys.argv[1] == 'DartHexapod-v1':
        rendergroup = [[0,1,2, 3,4,5], [6,7,8, 9,10,11], [12,13,14, 15,16,17]]
        titles = ['hind legs', 'middle legs', 'front legs']
        for i,rg in enumerate(rendergroup):
            plt.figure()
            plt.title(titles[i])
            for i in rg:
                plt.plot(np.array(actions)[:, i])
    plt.figure()
    plt.title('rewards')
    plt.plot(rew_seq, label='total rew')
    plt.plot(action_pen, label='action pen')
    plt.plot(vel_rew, label='vel rew')
    plt.plot(deviation_pen, label='dev pen')
    plt.legend()
    plt.figure()
    plt.title('com z')
    plt.plot(com_z)
    plt.figure()
    plt.title('x vel')
    plt.plot(x_vel)
    foot_contacts = np.array(foot_contacts)
    plt.figure()
    plt.title('foot contacts')
    plt.plot(1-foot_contacts[:, 0])
    plt.plot(1-foot_contacts[:, 1])
    plt.figure()

    if len(contact_force) > 0:
        plt.title('contact_force')
        plt.plot(np.array(contact_force)[:,0], label='x')
        plt.plot(np.array(contact_force)[:,1], label='y')
        plt.plot(np.array(contact_force)[:,2], label='z')
        plt.legend()
    plt.figure()
    plt.title('ref_rewards')
    plt.plot(ref_rewards)
    plt.figure()
    plt.title('ref_feat_rew')
    plt.plot(ref_feat_rew)
    plt.figure()
    plt.title('average velocity')
    plt.plot(avg_vels)
    print('total ref rewards ', np.sum(ref_rewards))
    print('total vel rewrads ', np.sum(vel_rew))
    print('total action rewards ', np.sum(action_pen))


    ################ save average action signals #################
    avg_action = np.mean(np.abs(actions), axis=1)
    np.savetxt('data/force_data/action_mean.txt', avg_action)
    np.savetxt('data/force_data/action_std.txt', np.std(np.abs(actions), axis=1))

    plt.show()





