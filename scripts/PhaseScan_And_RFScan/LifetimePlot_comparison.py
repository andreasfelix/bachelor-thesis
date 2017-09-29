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

directory = 'data_04_08_2017/LtCavs_csv/'

for j, filepath in enumerate(os.listdir(directory)):

    filepath = directory + filepath

    data = np.loadtxt(filepath, usecols=(6, 7), delimiter=",", skiprows=1)

    index_array = np.argsort(data[:, 1])
    voltage = data[:, 1][index_array] / 1000  # MV
    lifetime = data[:, 0][index_array]

    # mask data
    voltage = voltage[(lifetime > 0) & (lifetime < 2.0)]
    lifetime = lifetime[(lifetime > 0) & (lifetime < 2.0)]

    voltagenew, unique_counts = np.unique(voltage, return_counts=True)
    lifetimenew = np.zeros(len(voltagenew))
    std = np.zeros(len(voltagenew))

    print(voltagenew)

    tmp = 0
    for i, count in enumerate(unique_counts):
        lifetimenew[i] = np.mean(lifetime[tmp:tmp + count])
        std[i] = np.std(lifetime[tmp:tmp + count])
        tmp += count
    print(lifetimenew)

    print(os.path.basename(filepath).split(".")[0])
    # ax.plot(voltage, lifetime, 'o', lw=2, color=cmap(j / 9), label=os.path.basename(filepath).split(".")[0][17:-10])
    ax.errorbar(voltagenew, lifetimenew, yerr=std, color=cmap(j / 9), label=os.path.basename(filepath).split(".")[0][17:-7], capsize=3)

ax.set_xlabel('cavity voltage / MV')
ax.set_ylabel('Lifetime / h')
plt.xlim((0.5, 2.0))
plt.ylim((0.0, 2.0))
plt.legend(loc='lower right')
plt.tight_layout()

# print('images/{}.pdf'.format(os.path.basename(filepath).split(".")[0]))
plt.savefig('../../images/05-comparison-lifetime.pdf')
