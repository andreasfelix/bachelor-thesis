# -*- coding: utf-8 -*-
from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib as mpl

print('Matplotlib version: {}'.format(mpl.__version__))
SIZE_1 = 8
SIZE_2 = 10
SIZE_3 = 8

plt.rc('font', size=SIZE_1)  # controls default text sizes
plt.rc('axes', titlesize=SIZE_1)  # fontsize of the axes title
plt.rc('xtick', labelsize=SIZE_1)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE_1)  # fontsize of the tick labels
plt.rc('axes', labelsize=SIZE_2)  # fontsize of the x and y labels
plt.rc('legend', fontsize=SIZE_3)  # legend fontsize
cmap = mpl.cm.Set1
from matplotlib import rc

# # for Palatino and other serif fonts use:
# rc('font', **{'family': 'serif', 'serif': ['Palatino']})
# rc('text', usetex=True)

import numpy as np

pos_1_75GHz = np.array([0.22321, 0.30921, 0.39521, 0.48121, 0.56721])
pos_1_50GHz = np.array([1.05292, 1.15326, 1.2536, 1.35394, 1.45428])


def return_averg_worst(positons, steps=500):
    worst_arr = np.empty(steps)
    averg_arr = np.empty(steps)
    betamin_arr = np.linspace(0.01, 6, steps)
    for i, betamin in enumerate(betamin_arr):
        tmp = betamin + positons ** 2 / betamin
        worst_arr[i] = np.max(tmp)
        averg_arr[i] = np.mean(tmp)
    return betamin_arr, averg_arr, worst_arr


betamin_arr, averg_arr_1_75GHz, worst_arr_1_75GHz = return_averg_worst(pos_1_75GHz)
betamin_arr, averg_arr_1_50GHz, worst_arr_1_50GHz = return_averg_worst(pos_1_50GHz)

fig = plt.figure(figsize=(5, 3.97))

plt.plot(betamin_arr, worst_arr_1_75GHz, '-', c=cmap(0 / 9), label="1.75 GHz max beta", lw=2)
plt.plot(betamin_arr, averg_arr_1_75GHz, '--', c=cmap(0 / 9), label="1.75 GHz average beta", lw=2)
plt.plot(betamin_arr, worst_arr_1_50GHz, '-', c=cmap(1 / 9), label='1.50 GHz max beta', lw=2)
plt.plot(betamin_arr, averg_arr_1_50GHz, '--', c=cmap(1 / 9), label='1.50 GHz average beta', lw=2)
plt.plot([0, 10], [4, 4], '--k', lw=2)
plt.xlim(0, betamin_arr[-1])
plt.ylim(0, 6)

plt.grid(ls='dashed', lw=1)
plt.legend()

plt.xlabel("Minimum beta function $\\beta^*$ / m")
plt.ylabel("Maximum and average $\\beta$ / m")

plt.tight_layout()
plt.savefig("../../images/04-optimal-beta.pdf", transparent=True)
plt.show()