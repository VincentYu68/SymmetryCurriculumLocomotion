__author__ = 'yuwenhao'

import gym
import sys, os, time

import joblib
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json

np.random.seed(1)

if __name__ == '__main__':
    algnames = []
    for i in range(1, len(sys.argv)):
        algnames.append(sys.argv[i])

    all_data = []
    all_len = []
    for i in range(len(algnames)):
        all_data.append([])
        all_len.append([])
    for i, algname in enumerate(algnames):
        guess_names = [algname]
        for sd in range(20):
            guess_names.append(algname+str(sd))
        for name in guess_names:
            if os.path.exists(name):
                with open(name+'/progress.json') as data_file:
                    data = data_file.readlines()
                all_data[i].append([])
                all_len[i].append([])
                for line in data:
                    pline = json.loads(line.strip())
                    if 'EpRewMean' in pline:
                        all_data[i][-1].append(pline['EpRewMean'])
                    elif 'rollout/return' in pline:
                        all_data[i][-1].append(pline['rollout/return'])
                    else:
                        print('No return data available')
                    if 'EpLenMean' in pline:
                        all_len[i][-1].append(pline['EpLenMean'])

    colors = ['r','g','b','c','y']
    plt.figure()
    for gp in range(len(all_data)):
        for sp in range(len(all_data[gp])):
            if sp == 0:
                plt.plot(all_data[gp][sp], colors[gp], label=algnames[gp])
            else:
                plt.plot(all_data[gp][sp], colors[gp])
    plt.legend()

    plt.figure()
    for gp in range(len(all_len)):
        for sp in range(len(all_len[gp])):
            if sp == 0:
                plt.plot(all_len[gp][sp], colors[gp], label=algnames[gp])
            else:
                plt.plot(all_len[gp][sp], colors[gp])
    plt.legend()
    plt.show()








