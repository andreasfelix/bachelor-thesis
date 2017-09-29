# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

print('Matplotlib version: {}'.format(mpl.__version__))
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize

from matplotlib import rc

cmap = mpl.cm.Set1

# # for Palatino and other serif fonts use:
# rc('font', **{'family': 'serif', 'serif': ['Palatino']})
# rc('text', usetex=True)

import numpy
import sys
from os.path import expanduser
import time

sys.path.append(expanduser("~") + "/git")
import shutil
import pickle

if __name__ == '__main__':
    # setup
    fig = plt.figure('Necktieplot', figsize=(8, 12), facecolor='1', frameon=False)
    ms = 16  # markersize

    # bessy Q3
    with open("../../../data/bessy2_Q3_vs_Q5.pkl", "rb") as inputfile:
        pickle_dict = pickle.load(inputfile)

    grid = pickle_dict['grid']
    k_values_1 = pickle_dict['k_values_1']
    k_values_2 = pickle_dict['k_values_2']

    z = np.ma.masked_array(grid, mask=grid == 0)

    # plot your masked array
    ax = fig.add_subplot(211)
    props = dict(boxstyle='round', facecolor='white')
    plt.gca().text(0.05, 0.95, "1", transform=plt.gca().transAxes, fontsize=26, verticalalignment='top', bbox=props)

    plt.contourf(k_values_1, k_values_2, z, colors='white', antialiased=True)
    plt.contour(k_values_1, k_values_2, grid, colors='black', levels=[0.999], antialiased=True)

    # print(grid)
    # print(z)

    plt.xlabel("$-k_{Q5T2}$")
    plt.ylabel("$-k_{Q3T2}$")

    plt.plot(2.58798946, 2.45516822, "X", markersize=ms, color=cmap(0 / 9))
    plt.plot(2.58798946, 2.8, "X", markersize=ms, color=cmap(1 / 9))

    plt.xlim(0, 4)
    plt.ylim(0, 4)

    p = mpl.patches.Rectangle((0, 0), 4, 4, hatch='/', fill=None, zorder=0)
    ax.add_patch(p)

    # bessy2 Q4
    with open("../../../data/bessy2_Q4_vs_Q5.pkl", "rb") as inputfile:
        pickle_dict = pickle.load(inputfile)

    grid = pickle_dict['grid']
    k_values_1 = pickle_dict['k_values_1']
    k_values_2 = pickle_dict['k_values_2']

    z = np.ma.masked_array(grid, mask=grid == 0)

    ax2 = fig.add_subplot(212)
    props = dict(boxstyle='round', facecolor='white')
    plt.gca().text(0.05, 0.95, "2", transform=plt.gca().transAxes, fontsize=26, verticalalignment='top', bbox=props)

    plt.contourf(k_values_1, k_values_2, z, colors='white', antialiased=True)
    plt.contour(k_values_1, k_values_2, grid, colors='black', levels=[0.999], antialiased=True)

    # print(grid)
    # print(z)

    plt.xlabel("$-k_{Q5T2}$")
    plt.ylabel("$k_{Q4T2}$")

    plt.plot(2.58798946, 2.57898949, "X", markersize=ms, color=cmap(0 / 9))

    plt.xlim(0, 4)
    plt.ylim(0, 4)

    p = mpl.patches.Rectangle((0, 0), 4, 4, hatch='/', fill=None, zorder=0)
    plt.gca().add_patch(p)

    plt.tight_layout()
    plt.savefig("../../images/05-stability-all_def.pdf", transparent=True)
    # plt.savefig("../../images/05-stability-all.png", dpi=300)
    # plt.show()
