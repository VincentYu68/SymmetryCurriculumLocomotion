__author__ = 'yuwenhao'

import gym
import sys, os, time

import joblib
import numpy as np

import matplotlib.pyplot as plt
import json

np.random.seed(1)

if __name__ == '__main__':
    # setup for walker walking
    plot_setup = [
        [['ppo_DartWalker3d-v1101_energy04_vel1_1s_mirror4_velrew3_ab4_anklesprint100_5_rotpen0_rew05xinit_stagedcurriculum4s75s34ratio'], 'ECL + MSL'],
        #[['ppo_DartWalker3d-v199_energy04_vel1_1s_mirrorreal4_velrew3_ab4_asinput_damping5_torque1x_anklesprint100_5_rotpen01_rew01xinit_1kassistance', 'ppo_curriculum_150eachit_vel1_runningavg3_e04_DartWalker3d-v1_99_0.8_0.6_2500'], 'Learner-Centered CL + MSL'],
        [['ppo_DartWalker3d-v195_energy04_vel1_1s_mirror0_ab4_velrew3_damping5_anklesprint100_5_rotpen0_rew01xinit_stagedcurriculum4s75s34ratio'], 'ECL'],
        [['ppo_DartWalker3d-v198_energy04_vel1_1s_mirrorreal4_velrew3_ab4_asinput_damping5_torque1x_anklesprint100_5_rotpen01_rew01xinit'], 'MSL'],
        [['ppo_DartWalker3d-v196_energy04_vel1_1s_mirror4_velrew3_ab4_asinput_damping5_torque1x_anklesprint100_5_rotpen01_rew01xinit'], 'PPO'],
        #[['ppo_DartWalker3d-v1107_energy01_vel1_1s_mirror0_velrew3_asinput_damping5_torque1x_anklesprint0_ab4_rotpen0_rew01xinit_nocurriculum_baseline'], 'PPO high torque']
    ]

    # setup for walker running
    '''plot_setup = [
        [['ppo_DartWalker3d-v1106_energy03_vel5_3s_mirror4_velrew3_damping5_anklesprint100_ab7_rotpen0_rew01xinit_stagedcurriculum4s75s12ratio_07rewthres'], 'ECL + MSL'],
        [['ppo_DartWalker3d-v194_energy03_vel5_3s_mirror0_velrew3_damping7_anklesprint100_5_rotpen0_rew01xinit_stagedcurriculum4s75s34ratio'], 'ECL'],
        [['ppo_DartWalker3d-v197_energy03_vel5_3s_mirror4_velrew3_asinput_damping5_ab7_torque1x_anklesprint100_5_rotpen01_rew01xinit'], 'MSL'],
        [['ppo_DartWalker3d-v1100_energy03_vel5_3s_mirror0_velrew3_asinput_damping5_ab7_torque1x_anklesprint100_5_rotpen01_rew01xinit'], 'PPO']
    ]'''

    # setup for humanoid walking
    '''plot_setup = [
        [['ppo_DartHumanWalker-v1377_energy03_vel15_15s_mirror4_velrew3_ab5_norotpen_dofpen11508_rew05xinit_thigh160_50springankle_stagedcurriculum_075reduce_07rewthres_2kassist'], 'ECL + MSL'],
        [['ppo_DartHumanWalker-v1379_energy03_vel15_15s_mirror0_velrew3_ab5_norotpen_dofpen11508_rew05xinit_thigh160_50springankle_stagedcurriculum_075reduce_07rewthres_2kassist'], 'ECL'],
        [['ppo_DartHumanWalker-v1384_energy03_vel15_1s_mirror4_velrew3_dofpenxyz11515_rew05xinit_ab6_thigh160waist150shoulder100_damping5_50springankle_nocurriculum_baseline'], 'MSL'],
        [['ppo_DartHumanWalker-v1385_energy03_vel15_1s_mirror0_velrew3_dofpenxyz11515_rew05xinit_ab6_thigh160waist150shoulder100_damping5_50springankle_nocurriculum_baseline'], 'PPO']
    ]'''

    # setup for humanoid running
    '''plot_setup = [
        [['ppo_DartHumanWalker-v1395_energy015_vel5_3s_mirror4_velrew3_dofpenxyz11508_rew05xinit_ab7_thigh160waist150shoulder100_damping5_50springankle_stagedcurriculum_075reduce_07rewthres_1p2termination_2kassist'], 'ECL + MSL'],
        [['ppo_DartHumanWalker-v1380_energy015_vel5_3s_mirror0_velrew4_dofpenxyz11508_rew05xinit_ab9_thigh160waist150shoulder100_damping5_50springankle_stagedcurriculum_075reduce_07rewthres_2kassist',
          'ppo_DartHumanWalker-v1380_energy015_vel5_3s_mirror0_velrew4_dofpenxyz11508_rew05xinit_ab9_thigh160waist150shoulder100_damping5_50springankle_stagedcurriculum_075reduce_07rewthres_2kassist_cont',
          'ppo_DartHumanWalker-v1380_energy015_vel5_3s_mirror0_velrew3_dofpenxyz11508_rew05xinit_ab9_thigh160waist150shoulder100_damping5_50springankle_stagedcurriculum_075reduce_07rewthres_1p2termination_2kassist_contcont'], 'ECL'],
        [['ppo_DartHumanWalker-v1382_energy15_vel5_3s_mirror4_up03fwd03ltl15_spinepen1yaw001_thighyawpen005_velrewavg4_2s_dcon1_damping2kneethigh_thigh160knee100waist150_shoulder100_dqpen0_anklespring50_velrew05xinit_ab9_baseline'], 'MSL'],
        [['ppo_DartHumanWalker-v1383_energy15_vel5_3s_mirror0_up03fwd03ltl15_spinepen1yaw001_thighyawpen005_velrewavg4_2s_dcon1_damping2kneethigh_thigh160knee100waist150_shoulder100_dqpen0_anklespring50_velrew05xinit_ab9_baseline'], 'PPO']
    ]'''



    all_data = []
    legend_names = []
    for i, one_trial in enumerate(plot_setup):
        legend_names.append(one_trial[1])
        trial_names = one_trial[0]
        one_data = []
        for i, name in enumerate(trial_names):
            filepath = 'data/' + name
            if os.path.exists(filepath):
                with open(filepath+'/progress.json') as data_file:
                    data = data_file.readlines()
                for line in data:
                    pline = json.loads(line.strip())
                    one_data.append(pline['EpRewMean'])
        all_data.append(one_data)

    colors = ['g','r','b','c','y']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for sp in range(len(all_data)):
        ax.plot(all_data[sp], colors[sp], linewidth=2, label=legend_names[sp])
    plt.legend()

    plt.title('Simp Biped walking', fontsize=14)

    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Average Return", fontsize=14)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(13)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(13)

    plt.show()









