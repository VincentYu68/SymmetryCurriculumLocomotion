__author__ = 'yuwenhao'

import gym
import sys, os, time

import joblib
import numpy as np

import matplotlib.pyplot as plt
import json

np.random.seed(1)

if __name__ == '__main__':
    ht_mean = np.loadtxt('data/force_data/action_mean_ht.txt')
    ht_std = np.loadtxt('data/force_data/action_std_ht.txt')
    our_mean = np.loadtxt('data/force_data/action_mean_our.txt')
    our_std = np.loadtxt('data/force_data/action_std_our.txt')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(our_mean, 'g', label = 'ECL + MSL')
    ax.plot(ht_mean, 'y', label = 'PPO high torque')
    plt.legend()

    plt.title('Comparison on average action', fontsize=14)

    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("Average action", fontsize=14)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(13)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(13)

    plt.show()









