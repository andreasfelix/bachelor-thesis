# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg

print('Matplotlib version: {}'.format(mpl.__version__))
SMALL_SIZE = 18
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
cmap = mpl.cm.Set1
from matplotlib import rc

# # for Palatino and other serif fonts use:
# rc('font', **{'family': 'serif', 'serif': ['Palatino']})
# rc('text', usetex=True)

import numpy
from os.path import expanduser
import time
import sys

sys.path.append(expanduser("~") + "/git")

from element.pyfiles.compute.transfermatrix_element_slicing import returnlatticedata
from element.pyfiles.compute.twissdata import returntwissdata
from element.pyfiles.visualize.latticeplot import paintlattice, annotateline

# plot settings
fig = plt.figure(figsize=(20, 7), facecolor='white')
ylim_min = -2
ylim_max = 28
start_sec = np.array((142.5, 157.5))
start_pos = start_sec - 0.5
end_sec = np.array((157.5, 172.5))
end_pos = end_sec + 0.5

# plot
activelattice = expanduser("~") + "/git/element/lattices/standard/BII_2017-03-28_17-54_LOCOFitByPS_noID_ActualUserMode_third_best.lte"

latticedata = returnlatticedata(activelattice, 'felix_full')
twissdata = returntwissdata(latticedata)

# subplots
for i in range(2):
    ax = fig.add_subplot(1, 2, i + 1)

    # plot data
    plt.plot(twissdata.Cs, twissdata.betax, color=cmap(0 / 9), linewidth=2, label="$\\beta_x$ / m")
    plt.plot(twissdata.Cs, twissdata.betay, color=cmap(1 / 9), linewidth=2, label="$\\beta_y$ / m")
    plt.plot(twissdata.Cs, twissdata.etax * 10, color=cmap(2 / 9), linewidth=2, label="$\\eta_x$ / cm")

    # ticks
    plt.gca().xaxis.grid(which='minor', linestyle='dashed')
    plt.gca().set_xticks(np.linspace(latticedata.LatticeLength / 32, latticedata.LatticeLength * (1 - 1 / 32), 16, endpoint=True), minor=True)
    plt.gca().set_xticks(np.linspace(start_sec[i], end_sec[i], 7, endpoint=True))
    plt.gca().yaxis.grid(linestyle='dashed')
    plt.gca().set_xlabel('orbit position $s$ / m')
    plt.gca().set_yticks(np.linspace(0, 28, 8, endpoint=True))

    # limits
    plt.xlim(start_pos[i], end_pos[i])
    plt.ylim(ylim_min, ylim_max)
    paintlattice(start_pos[i], end_pos[i], latticedata, ylim_min, ylim_max, halfsize=True, noborder=True)
    if i == 0:
        plt.annotate("D6", xy=(start_pos[i] + 8, 0.81 * ylim_max), fontsize=20, va='center', ha='center', clip_on=True, zorder=102, weight='bold')
        plt.annotate("UE56+U139 (femto slicing)", xy=(start_pos[i] + 8, 0.745 * ylim_max), fontsize=20, va='center', ha='center', clip_on=True, zorder=102)
    else:
        plt.annotate("T6", xy=(start_pos[i] + 8, 0.81 * ylim_max), fontsize=20, va='center', ha='center', clip_on=True, zorder=102, weight='bold')
        plt.annotate("CPMU17 + UE48 (EMIL)", xy=(start_pos[i] + 8, 0.745 * ylim_max), fontsize=20, va='center', ha='center', clip_on=True, zorder=102)

# annotate
start = 0.22
gap = 0.10
fs = 24
height_1 = 0.97
height_legend = height_1

# annotate active lattice
plt.annotate("standard 2017", xy=(0.98, height_1), xycoords='figure fraction', va='center', ha='right', fontsize=fs - 5, weight='bold')
annolist_string = "$Q_x$: {:.2f} ({:.0f} kHz)   $Q_y$: {:.2f} ({:.0f} kHz)   $\\alpha_C$: {:.2e}".format(twissdata.Qx, twissdata.QxFreq, twissdata.Qy, twissdata.QyFreq, twissdata.alphac)
plt.annotate(annolist_string, xy=(start, height_1), xycoords='figure fraction', va='center', ha='left', fontsize=fs - 4)

# annotate beta eta
plt.annotate('$\\beta_x$/m', xy=(0.05, height_legend), color=mpl.cm.Set1(0 / 9), xycoords='figure fraction', va='center', ha='right', fontsize=fs)
plt.annotate('$\\beta_y$/m', xy=(0.10, height_legend), color=mpl.cm.Set1(1 / 9), xycoords='figure fraction', va='center', ha='right', fontsize=fs)
plt.annotate('$\\eta_x$/10cm', xy=(0.18, height_legend), color=mpl.cm.Set1(2 / 9), xycoords='figure fraction', va='center', ha='right', fontsize=fs)

# save fig
plt.tight_layout()

plt.gcf().subplots_adjust(top=0.88)
plt.savefig("../../images/04-emil-femto-slicing.pdf", transparent=True)
