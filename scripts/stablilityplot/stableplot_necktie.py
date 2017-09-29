# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

print('Matplotlib version: {}'.format(mpl.__version__))
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize

from matplotlib import rc

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
    with open("../../../data/necktiedata_2000.pkl", "rb") as inputfile:
        pickle_dict = pickle.load(inputfile)

    grid = pickle_dict['grid']
    k_values_1 = pickle_dict['k_values_1']
    k_values_2 = pickle_dict['k_values_2']

    z = np.ma.masked_array(grid, mask=grid == 0)

    # plot your masked array
    fig = plt.figure('Necktieplot', figsize=(10, 8), facecolor='1', frameon=False)
    ax = fig.add_subplot(111)

    plt.contourf(k_values_1, k_values_2, z, colors='white', antialiased=True)
    plt.contour(k_values_1, k_values_2, grid, colors='black', levels=[0.999], antialiased=True)

    # print(grid)
    # print(z)

    plt.xlabel("$k_{QF}$")
    plt.ylabel("-$k_{QD}$")

    plt.xlim(0, 0.3)
    plt.ylim(0, 0.3)
    ms = 18
    cmap = mpl.cm.Set1
    plt.plot(0.02, 0.02, "X", label="stable (1)", markersize=ms, color=cmap(0 / 9))
    plt.plot(0.2338, 0.2338, "X", label="stable (2)", markersize=ms, color=cmap(2 / 9))
    plt.plot(0.02, 0.07, "X", label="not stable (3)", markersize=ms, color=cmap(1 / 9))
    plt.plot(0.26, 0.26, "X", label="not stable (4)", markersize=ms, color=cmap(3 / 9))

    plt.legend(fontsize=24)
    plt.tight_layout()

    p = mpl.patches.Rectangle((0, 0), 2, 2, hatch='/', fill=None, zorder=-10)
    ax.add_patch(p)
    plt.savefig("../../images/05-necktie-plot.pdf", transparent=True)
    # plt.show()
