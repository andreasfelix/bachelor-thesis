# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg

print('Matplotlib version: {}'.format(mpl.__version__))
SMALL_SIZE = 14
MEDIUM_SIZE = 14
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

directory = 'data_04_08_2017/data_csv/'
for filepath in os.listdir(directory):
    filepath = directory + filepath

    data = np.loadtxt(filepath, usecols=(1,5), delimiter=",", skiprows=1)

    efficiency = data[:,0]
    phase = data[:,1]

    fig = plt.figure(facecolor='white', figsize=(16, 9))
    ax = fig.add_subplot(111)

    plt.plot(phase,efficiency,'o', lw=2)
    plt.plot([0.0,2.0], [90,90], 'k--', lw=2)
    ax.set_xlabel('Phase sy-sr in ns', fontsize=14)
    ax.set_ylabel('Injection Efficiency in %', fontsize=14)
    plt.gca().tick_params(direction='in')
    plt.tight_layout()
    plt.savefig('images/{}.pdf'.format(os.path.basename(filepath).split(".")[0]))
