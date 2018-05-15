__author__ = 'yuwenhao'

import gym
import sys, os, time

import joblib
import numpy as np

import matplotlib.pyplot as plt
import json
import re

np.random.seed(1)

if __name__ == '__main__':
    ############################## OLD APPROACH ##################################
    # walking test
    #basepolicy = 'data/ppo_DartWalker3d-v199_energy04_vel1_1s_mirrorreal4_velrew3_ab4_asinput_damping5_torque1x_anklesprint100_5_rotpen01_rew01xinit_1kassistance/'
    #directory = 'data/ppo_curriculum_150eachit_vel1_runningavg3_e04_DartWalker3d-v1_99_0.8_0.6_2500/'

    # running test
    basepolicy = 'data/ppo_DartWalker3d-v193_energy03_vel5_3s_mirror4_velrew3_asinput_damping5_torque1x_anklesprint100_5_ab7_rotpen0_rew01xinit/'
    directory = 'data/ppo_curriculum_150eachit_vel5_3s_runningavg3_e03_DartWalker3d-v1_0_0.8_0.6_2500/'

    learning_curve = []
    with open(basepolicy + '/progress.json') as data_file:
        data = data_file.readlines()
        for line in data:
            pline = json.loads(line.strip())
            learning_curve.append(pline['EpRewMean'])
    learning_curve = learning_curve[0:250]
    with open(directory + '/progress.json') as data_file:
        data = data_file.readlines()
        for line in data:
            pline = json.loads(line.strip())
            learning_curve.append(pline['EpRewMean'])
    learning_curve = np.array(learning_curve)

    curriculum_list = []
    iter_list = {}
    for fname in os.listdir(directory):
        if 'policy_params_' in fname:
            split_name = re.split(r'[\[\];,\s]\s*', fname)
            cur_key = [float(split_name[1]), float(split_name[2])*2]
            if cur_key not in curriculum_list:
                curriculum_list.append(cur_key)

            split_it = re.split(r'[\[\]_.;,\s]\s*', split_name[-1])
            if str(cur_key) not in iter_list:
                iter_list[str(cur_key)] = int(split_it[1])
            else:
                iter_list[str(cur_key)] = max(iter_list[str(cur_key)], int(split_it[1]))
    curriculum_list.sort(reverse=True)

    distance_metric = []
    pretrain_iter = 250
    accum_iter = pretrain_iter
    iteration_list = []

    for curr in curriculum_list:
        distance_metric.append(np.linalg.norm(curr))
        accum_iter += iter_list[str(curr)] + np.random.randint(0, 10)
        iteration_list.append(accum_iter)

    iteration_list.insert(0, 0)
    distance_metric.insert(0, distance_metric[0])
    distance_metric = np.array(distance_metric)# / distance_metric[0]



    ############################### NEW APPROACH #####################
    # walking learning
    #env_cent_directory = 'data/ppo_DartWalker3d-v1101_energy04_vel1_1s_mirror4_velrew3_ab4_anklesprint100_5_rotpen0_rew05xinit_stagedcurriculum4s75s34ratio/'

    # running learning
    env_cent_directory = 'data/ppo_DartWalker3d-v1106_energy03_vel5_3s_mirror4_velrew3_damping5_anklesprint100_ab7_rotpen0_rew01xinit_stagedcurriculum4s75s12ratio_07rewthres/'

    envcent_learning_curve = []
    with open(env_cent_directory + '/progress.json') as data_file:
        data = data_file.readlines()
        for line in data:
            pline = json.loads(line.strip())
            envcent_learning_curve.append(pline['EpRewMean'])

    envcentcurriculum_list = []
    envcenteriter_list = {}
    for fname in os.listdir(env_cent_directory):
        if '0.0' in fname:
            split_name = re.split(r'[\[\];,\s]\s*', fname)
            cur_key = [float(split_name[4]), float(split_name[4])]
            if cur_key not in envcentcurriculum_list:
                envcentcurriculum_list.append(cur_key)

                envcenteriter_list[str(cur_key)] = 10*(len(os.listdir(env_cent_directory + fname))-1)+1
        envcentcurriculum_list.sort(reverse=True)

    envcentdistance_metric = []
    accum_iter = 0
    envcenteriteration_list = []


    for curr in envcentcurriculum_list:
        envcentdistance_metric.append(np.linalg.norm(curr))
        accum_iter += envcenteriter_list[str(curr)] + np.random.randint(0, 10)
        envcenteriteration_list.append(accum_iter)
    envcenteriteration_list.insert(0, 0)
    envcentdistance_metric.insert(0, envcentdistance_metric[0])

    envcentdistance_metric = np.array(envcentdistance_metric)# / envcentdistance_metric[0]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(envcenteriteration_list, envcentdistance_metric, linewidth=2, label = 'Env-Cent Learning')
    ax.plot(iteration_list, distance_metric, color='g', linewidth=2, label = 'Learner-Cent Learning')
    plt.legend()

    plt.title('Curriculum Progress', fontsize=14)

    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Curriculum Progress", fontsize=14)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(13)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(13)



    ###################### plot learning curve #############################

    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    ax.plot(envcent_learning_curve, linewidth=2, label='Env-Cent Learning')
    ax.plot(learning_curve[0:iteration_list[-1]], color='g', linewidth=2, label='Learner-Cent Learning')
    plt.legend()

    plt.title('Learning Curve', fontsize=14)

    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Average Return", fontsize=14)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(13)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(13)


    ##################### plot curriculum path #############################

    fig3 = plt.figure()

    ax = fig3.add_subplot(1, 1, 1)
    envcentcurriculum_list = np.array(envcentcurriculum_list)
    curriculum_list = np.array(curriculum_list)
    ax.plot(envcentcurriculum_list[:,0], envcentcurriculum_list[:,1], '*', linewidth=2, label='Env-Cent Learning')
    ax.plot(curriculum_list[:,0], curriculum_list[:,1], '+g', linewidth=2, label='Learner-Cent Learning')
    plt.legend()

    plt.title('Curriculum Path', fontsize=14)

    plt.xlabel("kp", fontsize=14)
    plt.ylabel("kd", fontsize=14)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(13)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(13)

    plt.show()







