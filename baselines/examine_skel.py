__author__ = 'yuwenhao'

import gym
import sys, os, time

import joblib
import numpy as np

import matplotlib.pyplot as plt
from gym import wrappers

np.random.seed(15)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        env = gym.make(sys.argv[1])
    else:
        env = gym.make('DartHumanWalker-v1')

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

    env.env.dart_world.set_gravity(np.zeros(3))

    q = env.env.robot_skeleton.q

    cur_dof = 0
    t_inc = 0.03
    cur_val = 0.5
    T = 0.0
    while True:
        if env.env.robot_skeleton.q_lower[cur_dof] < -100:
            lval = -1.0
        else:
            lval = env.env.robot_skeleton.q_lower[cur_dof]
        if env.env.robot_skeleton.q_upper[cur_dof] > 100:
            hval = 1.0
        else:
            hval = env.env.robot_skeleton.q_upper[cur_dof]
        qval = (hval - lval) * cur_val + lval

        cur_q = env.env.robot_skeleton.q
        cur_q[cur_dof] = qval
        env.env.robot_skeleton.q = cur_q

        cur_val += t_inc
        if cur_val >= 1.0 or cur_val <= -0.0:
            t_inc *= -1

        T += np.abs(t_inc)
        if T > 2.0:
            cur_dof += 1
            T = 0.0
            if cur_dof >= len(cur_q):
                cur_dof = 0
            env.reset()
        env.render()

