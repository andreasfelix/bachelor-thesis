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

# # for Palatino and other serif fonts use:
# rc('font', **{'family': 'serif', 'serif': ['Palatino']})
# rc('text', usetex=True)

import os
import time
import sys

sys.path.append(os.path.expanduser("~") + "/git")

from element.pyfiles.compute.transfermatrix_element_slicing import returnlatticedata
from element.pyfiles.compute.twissdata import returntwissdata
from element.pyfiles.visualize.latticeplot import paintlattice, annotateline

# plot settings
latticelist = [os.path.expanduser("~") + "/git/element/lattices/standard/BII_2017-03-28_17-54_LOCOFitByPS_noID_ActualUserMode_third_best",
               os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/V1/V1",
               os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/V2/V2_2017-05-14_15-42-32",
               os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/V3/V3_2017-05-14_15-45-23",
               os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/V4/V4_2017-05-14_15-55-33",
               os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/V5/V5_2017-05-14_05-17-07",
               os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/Vall/Vall_2017-08-01_15-35-13"]
               # os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/V2Q3T/V2Q3T_2017-07-31_16-00-02",
               # os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/V2Q4T/V2Q4T_2017-07-31_15-31-53",
               # os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/V2Q5/V2Q5_2017-07-31_15-47-44",
               # os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/VOF/VOF_2017-08-01_15-02-23"]

latticenamelist = ["current", "V1", "V2", "V3", "V4", "V5", 'Vall', 'V2Q3T', 'V2Q4T', 'V2Q5', 'VOF']

num = len(latticelist)
fig = plt.figure(figsize=(16, 3 * num), facecolor='white')
row_length = 8
rows = row_length * num
ylim_min = -2
ylim_max = 32
start_section = 37.5
length = 15

# plot
for i in range(num):
    # design lattice
    activelattice = latticelist[i] + '.lte'
    latticedata = returnlatticedata(activelattice, 'felix_full')
    twissdata = returntwissdata(latticedata)

    plt.subplot2grid((rows, 1), (i * row_length + 1, 0), colspan=1, rowspan=row_length - 1)

    plt.plot(twissdata.Cs, twissdata.betax, color=cmap(0 / 9), linewidth=1.5, label="$\\beta_x$ / m")
    plt.plot(twissdata.Cs, twissdata.betay, color=cmap(1 / 9), linewidth=1.5, label="$\\beta_y$ / m")
    plt.plot(twissdata.Cs, twissdata.etax * 10, color=cmap(2 / 9), linewidth=1.5, label="$\\eta_x$ / cm")

    plt.xlim(0, latticedata.LatticeLength)
    plt.ylim(ylim_min, ylim_max)

    plt.gca().set_xlabel('orbit position $s$ / m')
    paintlattice(0, latticedata.LatticeLength, latticedata, ylim_min, ylim_max, halfsize=True, noborder=True)
    annotateline(latticedata, ylim_max)

    # ticks
    plt.gca().xaxis.grid(which='minor', linestyle='dotted')
    plt.gca().set_xticks(np.linspace(0, latticedata.LatticeLength, 17, endpoint=True))
    plt.gca().set_xticks(np.linspace(latticedata.LatticeLength / 32, latticedata.LatticeLength * (1 - 1 / 32), 16, endpoint=True), minor=True)
    plt.gca().yaxis.grid(alpha=0.5, zorder=0, linestyle='dotted')
    plt.gca().set_yticks(np.linspace(0, 32, 9, endpoint=True))

    # annotate
    start = 0.2
    gap = 0.10
    fs = 18
    offsett = 0.015
    height_1 = (1 - offsett) - (1 - offsett) / num * i
    height_legend = height_1

    # annotate active lattice
    plt.annotate(latticenamelist[i], xy=(0.98, height_1), xycoords='figure fraction', va='center', ha='right', fontsize=fs, weight = 'bold')
    annolist_string = "$Q_x$: {:.2f} ({:.0f} kHz)   $Q_y$: {:.2f} ({:.0f} kHz)   $\\alpha_C$: {:.2e}".format(twissdata.Qx, twissdata.QxFreq, twissdata.Qy, twissdata.QyFreq, twissdata.alphac)
    plt.annotate(annolist_string, xy=(start, height_1), xycoords='figure fraction', va='center', ha='left', fontsize=fs)

    # annotate beta eta
    plt.annotate('$\\beta_x$/m', xy=(0.05, height_legend), color=mpl.cm.Set1(0 / 9), xycoords='figure fraction', va='center', ha='right', fontsize=fs+2)
    plt.annotate('$\\beta_y$/m', xy=(0.1025, height_legend), color=mpl.cm.Set1(1 / 9), xycoords='figure fraction', va='center', ha='right', fontsize=fs+2)
    plt.annotate('$\\eta_x$/10cm', xy=(0.18, height_legend), color=mpl.cm.Set1(2 / 9), xycoords='figure fraction', va='center', ha='right', fontsize=fs+2)

    # draw section
    if i > 0 and i < 6:
        j = i - 1
        p = mpl.patches.Rectangle((start_section - j * length, ylim_min), (1 + 2 * j) * length, 50, ec="none", zorder=-10, alpha=0.15, color=cmap(1 / 9))
        plt.gca().add_patch(p)
        if i > 3:
            j = i - 4
            p = mpl.patches.Rectangle((latticedata.LatticeLength, ylim_min), - (0.5 + j) * length, 50, ec="none", zorder=-10, alpha=0.15, color=cmap(1 / 9))
            plt.gca().add_patch(p)
    if i == 6:
        p = mpl.patches.Rectangle((0, ylim_min), 16 * length, 50, ec="none", zorder=-10, alpha=0.15, color=cmap(1 / 9))
        plt.gca().add_patch(p)
# save fig
plt.tight_layout()
plt.savefig("../../images/05-V-versions-comparison.pdf")
