# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg

print('Matplotlib version: {}'.format(mpl.__version__))
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
cmap = mpl.cm.Set1
from matplotlib import rc
import os

# # for Palatino and other serif fonts use:
# rc('font', **{'family': 'serif', 'serif': ['Palatino']})
# rc('text', usetex=True)

fig = plt.figure(facecolor='white', figsize=(9, 5))
ax = fig.add_subplot(111)

directory = 'data_04_08_2017/data_csv/'
for j, filepath in enumerate(os.listdir(directory)):
    filepath = directory + filepath

    data = np.loadtxt(filepath, usecols=(1, 5), delimiter=",", skiprows=1)

    index_array = np.argsort(data[:, 1])
    phase = data[:, 1][index_array]
    efficiency = data[:, 0][index_array]

    phasenew, unique_counts = np.unique(phase, return_counts=True)
    efficiencynew = np.zeros(len(phasenew))

    tmp = 0
    for i, count in enumerate(unique_counts):
        efficiencynew[i] = np.mean(efficiency[tmp:tmp + count])
        tmp += count

    print(os.path.basename(filepath).split(".")[0])
    ax.plot(phasenew, efficiencynew, '-', lw=2, color=cmap(j / 9), label=os.path.basename(filepath).split(".")[0][17:-10])

# # V4 optics
# filepath = 'data_V4_mai/20170517_Q5offT2_V4_NLoptimised.str'
#
# data = np.loadtxt(filepath, usecols=(1,2), delimiter="\t", skiprows=1)
#
# efficiency = data[:,1]
# phase = data[:,0]
#
# phasenew, unique_counts = np.unique(phase, return_counts=True)
# efficiencynew = np.zeros(len(phasenew))
#
# tmp = 0
# for i, count in enumerate(unique_counts):
#     efficiencynew[i] = np.mean(efficiency[tmp:tmp + count])
#     tmp += count
#
# plt.plot(phasenew, efficiencynew,'--', lw=2, color=cmap(4 / 9),label="V4optics-04_08_2017")


plt.plot([0.0, 2.0], [90, 90], 'k--', lw=1.5)
ax.set_xlabel('Phase sy-sr / ns')
ax.set_ylabel('Injection Efficiency in %')
ax.set_ylim((-5, 100))
plt.legend(loc='lower right')
plt.xlim((0, 2.0))
plt.yticks(np.linspace(0, 100, 11))
plt.tight_layout()
# print('images/{}.pdf'.format(os.path.basename(filepath).split(".")[0]))
plt.savefig('../../images/05-comparison-phase-acceptance.pdf')
