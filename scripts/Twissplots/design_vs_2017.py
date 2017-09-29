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

import numpy
from os.path import expanduser
import time
import sys

sys.path.append(expanduser("~") + "/git")

from element.pyfiles.compute.transfermatrix_element_slicing import returnlatticedata
from element.pyfiles.compute.twissdata import returntwissdata
from element.pyfiles.visualize.latticeplot import paintlattice, annotateline

# plot settings
fig = plt.figure(figsize=(16, 7.5), facecolor='white')
rows = 16
ylim_min = -2
ylim_max = 30

# plot
latticelist = [expanduser("~") + "/git/element/lattices/standard/BII_2016-06-10_user_Sym_noID_DesignLattice1996.lte",
               expanduser("~") + "/git/element/lattices/standard/BII_2017-03-28_17-54_LOCOFitByPS_noID_ActualUserMode_third_best.lte"]
names = ['Design lattice 1996', 'Standard lattice 2017']

for i in range(2):
    # design lattice
    activelattice = latticelist[i]
    latticedata = returnlatticedata(activelattice, 'felix_full')
    twissdata = returntwissdata(latticedata)

    plt.subplot2grid((rows, 1), (i * int(rows / 2) + 1, 0), colspan=1, rowspan=int(rows / 2) - 1)

    plt.plot(twissdata.Cs, twissdata.betax, color=cmap(0 / 9), linewidth=1.5, label="$\\beta_x$ / m")
    plt.plot(twissdata.Cs, twissdata.betay, color=cmap(1 / 9), linewidth=1.5, label="$\\beta_y$ / m")
    plt.plot(twissdata.Cs, twissdata.etax * 10, color=cmap(2 / 9), linewidth=1.5, label="$\\eta_x$ / cm")

    plt.xlim(0, latticedata.LatticeLength)
    plt.ylim(ylim_min, ylim_max)

    paintlattice(0, latticedata.LatticeLength, latticedata, ylim_min, ylim_max, halfsize=True, noborder=True)
    annotateline(latticedata, ylim_max + 1)

    # ticks
    # plt.gca().get_xaxis().set_tick_params(direction='in')
    # plt.gca().get_yaxis().set_tick_params(direction='in')

    plt.gca().xaxis.grid(which='minor', linestyle='dotted')
    plt.gca().yaxis.grid(alpha=0.5, zorder=0, linestyle='dotted')
    plt.gca().set_xticks(np.linspace(0, latticedata.LatticeLength, 17, endpoint=True))
    plt.gca().set_xticks(np.linspace(latticedata.LatticeLength / 32, latticedata.LatticeLength * (1 - 1 / 32), 16, endpoint=True), minor=True)
    plt.gca().set_xlabel('orbit position $s$ / m')
    plt.gca().set_yticks(np.linspace(0, 28, 8, endpoint=True))

    # annotate
    start = 0.2
    gap = 0.10
    fs = 18
    height_1 = 0.97 - 0.5 * i
    height_legend = height_1
    print(twissdata.Qx, twissdata.Qy)

    # annotate active lattice
    plt.annotate(names[i], xy=(0.98, height_1), xycoords='figure fraction', va='center', ha='right', fontsize=fs - 4, weight='bold')
    annolist_string = "$Q_x$: {:.2f} ({:.0f} kHz)   $Q_y$: {:.2f} ({:.0f} kHz)   $\\alpha_C$: {:.2e}".format(twissdata.Qx, twissdata.QxFreq, twissdata.Qy, twissdata.QyFreq, twissdata.alphac)
    plt.annotate(annolist_string, xy=(start, height_1), xycoords='figure fraction', va='center', ha='left', fontsize=fs - 4)

    # annotate beta eta
    plt.annotate('$\\beta_x$/m', xy=(0.05, height_legend), color=mpl.cm.Set1(0 / 9), xycoords='figure fraction', va='center', ha='right', fontsize=fs)
    plt.annotate('$\\beta_y$/m', xy=(0.1025, height_legend), color=mpl.cm.Set1(1 / 9), xycoords='figure fraction', va='center', ha='right', fontsize=fs)
    plt.annotate('$\\eta_x$/10cm', xy=(0.18, height_legend), color=mpl.cm.Set1(2 / 9), xycoords='figure fraction', va='center', ha='right', fontsize=fs)

# save fig
plt.tight_layout()
plt.savefig("../../images/04-design-vs-2017-lattice.pdf")
plt.show()