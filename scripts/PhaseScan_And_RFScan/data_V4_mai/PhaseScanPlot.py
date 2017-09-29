# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg

print('Matplotlib version: {}'.format(mpl.__version__))
SMALL_SIZE = 24
MEDIUM_SIZE = 24
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize

cmap = mpl.cm.Set1
from matplotlib import rc
import os

# # for Palatino and other serif fonts use:
# rc('font', **{'family': 'serif', 'serif': ['Palatino']})
# rc('text', usetex=True)

filepath = '20170517_Q5offT2_V4_NLoptimised.str'

data = np.loadtxt(filepath, usecols=(1, 2), delimiter="\t", skiprows=1)

efficiency = data[:, 1]
phase = data[:, 0]

phasenew, unique_counts = np.unique(phase, return_counts=True)
efficiencynew = np.zeros(len(phasenew))
std = np.zeros(len(phasenew))

tmp = 0
for i, count in enumerate(unique_counts):
    efficiencynew[i] = np.mean(efficiency[tmp:tmp + count])
    tmp += count

fig = plt.figure(facecolor='white', figsize=(16, 9))
ax = fig.add_subplot(111)

plt.plot(phase, efficiency, 'o', lw=2)
plt.plot(phasenew, efficiencynew, '-r', lw=2)
plt.plot([0.0, 2.0], [90, 90], 'k--', lw=2)
ax.set_xlabel('Phase sy-sr / ns')
ax.set_ylabel('Injection Efficiency in %')
plt.yticks(np.linspace(0, 100, 11))
ax.set_xlim((0, phasenew[-1]))
plt.tight_layout()

plt.savefig('../../../images/05-Phase-acceptance-V4-mai.pdf')
