# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

print('Matplotlib version: {}'.format(mpl.__version__))

cmap = mpl.cm.Set1
SIZE_1 = 14
SIZE_2 = 16
SIZE_3 = 12

plt.rc('font', size=SIZE_1)  # controls default text sizes
plt.rc('axes', titlesize=SIZE_1)  # fontsize of the axes title
plt.rc('xtick', labelsize=SIZE_1)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE_1)  # fontsize of the tick labels
plt.rc('axes', labelsize=SIZE_2)  # fontsize of the x and y labels
plt.rc('legend', fontsize=SIZE_3)  # legend fontsize

cmap = mpl.cm.rainbow


import sys
from os.path import expanduser
import time

sys.path.append(expanduser("~") + "/git")
import shutil
import pickle

if __name__ == '__main__':
    with open("../../../data/bessy2_Q3_vs_Q4_Q5_off_betavalue.pkl", "rb") as inputfile:
        pickle_dict = pickle.load(inputfile)

    grid = pickle_dict['grid']
    k_values_1 = pickle_dict['k_values_1']
    k_values_2 = pickle_dict['k_values_2']

    z = np.ma.masked_array(grid, mask=grid == 0)

    # plot your masked array
    fig = plt.figure('Bessy2 stability', figsize=(8, 5), facecolor='1', frameon=False)
    ax = fig.add_subplot(111)

    levels = np.linspace(3, 12, 100)
    levels = levels ** 3
    levels = [20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 200, 300, 500, 1000, 2000, 10000]
    colors = [cmap(_ / len(levels)) for _ in range(len(levels))]

    cs = plt.contourf(k_values_1, k_values_2, z.T, colors=colors, levels=levels)
    fig.colorbar(cs, ax=ax, format="%.2f")
    # plt.contour(k_values_1, k_values_2, grid, colors='black', levels=[0.999], antialiased=True)


    plt.xlabel("-$k_{Q3T2}$")
    plt.ylabel("$k_{Q4T2}$")

    # plt.plot(2.45516822, 2.57898949, "Xr", markersize=18)
    # plt.plot(2.629593, 2.032261, "X", markersize=18, color=cmap(9 / 9))
    #
    # plt.xlim(1.3, 2.3)
    # plt.ylim(-4, 4)

    # p = mpl.patches.Rectangle((-4, -4), 8, 8, hatch='/', fill=None, zorder=-10)
    # ax.add_patch(p)
    plt.savefig("../../images/04-stability-plot-bessy2_Q3_vs_Q4_Q5_off_betamaxvalue.pdf")
    plt.show()
