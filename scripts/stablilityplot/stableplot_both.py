# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

print('Matplotlib version: {}'.format(mpl.__version__))
SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize

from matplotlib import rc

# for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

import numpy
import sys
from os.path import expanduser
import time

sys.path.append(expanduser("~") + "/git")
import shutil
import pickle

if __name__ == '__main__':
    fig = plt.figure('Necktieplot', figsize=(12, 5.5), facecolor='1', frameon=False)

    # necktie
    with open("../../../data/necktiedata_2000.pkl", "rb") as inputfile:
        pickle_dict = pickle.load(inputfile)

    grid = pickle_dict['grid']
    k_values_1 = pickle_dict['k_values_1']
    k_values_2 = pickle_dict['k_values_2']

    z = np.ma.masked_array(grid, mask=grid == 0)

    # plot your masked array
    ax = fig.add_subplot(121)

    plt.contourf(k_values_1, k_values_2, z, colors='white', antialiased=True)
    plt.contour(k_values_1, k_values_2, grid, colors='black', levels=[0.999], antialiased=True)

    # print(grid)
    # print(z)

    plt.xlabel("$k_{QF}$")
    plt.ylabel("-$k_{QD}$")

    plt.xlim(0, 0.3)
    plt.ylim(0, 0.3)

    p = mpl.patches.Rectangle((0, 0), 2, 2, hatch='/', fill=None, zorder=-10)
    ax.add_patch(p)

    # bessy2
    with open("../../../data/bessy2.pkl", "rb") as inputfile:
        pickle_dict = pickle.load(inputfile)

    grid = pickle_dict['grid']
    k_values_1 = pickle_dict['k_values_1']
    k_values_2 = pickle_dict['k_values_2']

    z = np.ma.masked_array(grid, mask=grid == 0)

    ax2 = fig.add_subplot(122)

    plt.contourf(k_values_1, k_values_2, z, colors='white', antialiased=True)
    plt.contour(k_values_1, k_values_2, grid, colors='black', levels=[0.999], antialiased=True)

    # print(grid)
    # print(z)

    plt.xlabel("-$k_{Q5T2}$")
    plt.ylabel("$k_{Q4T2}$")

    plt.plot(2.59997779, 2.57898949, "or", markersize=8)

    plt.xlim(0, 4)
    plt.ylim(0, 4)

    p = mpl.patches.Rectangle((0, 0), 5, 5, hatch='/', fill=None, zorder=-10)
    ax2.add_patch(p)

    plt.tight_layout()
    plt.savefig("../../images/04-stability-bessy2-necktie.pdf")
    # plt.show()
