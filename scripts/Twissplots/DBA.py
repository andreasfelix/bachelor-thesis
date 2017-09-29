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
fig = plt.figure(figsize=(16, 6), facecolor='white')
rows = 12
ylim_min = -2
ylim_max = 25
start_pos = [22, 37]
end_pos = [38, 53]
# plot
activelattice = expanduser("~") + "/git/element/lattices/BII_2016-06-10_user_Sym_noID_DesignLattice1996.lte"

latticedata = returnlatticedata(activelattice, 'felix_full')
twissdata = returntwissdata(latticedata)

for i in range(2):
    plt.subplot2grid((rows, 2), (1, i), colspan=1, rowspan=11)

    # plot data
    plt.plot(twissdata.Cs, twissdata.betax, color=cmap(0 / 9), linewidth=1.5, label="$\\beta_x$ / m")
    plt.plot(twissdata.Cs, twissdata.betay, color=cmap(1 / 9), linewidth=1.5, label="$\\beta_y$ / m")
    plt.plot(twissdata.Cs, twissdata.etax * 10, color=cmap(2 / 9), linewidth=1.5, label="$\\eta_x$ / cm")

    # ticks
    plt.gca().get_xaxis().set_tick_params(direction='in')
    plt.gca().get_yaxis().set_tick_params(direction='in')
    plt.gca().xaxis.grid(which='minor', linestyle='dashed')
    plt.gca().set_xticks(np.linspace(latticedata.LatticeLength / 32, latticedata.LatticeLength * (1 - 1 / 32), 16, endpoint=True), minor=True)
    plt.gca().yaxis.grid(linestyle='dashed')
    plt.gca().set_xlabel('orbit position $s$ / m')

    # limits
    plt.xlim(start_pos[i], end_pos[i])
    plt.ylim(ylim_min, ylim_max)
    paintlattice(start_pos[i], end_pos[i], latticedata, ylim_min, ylim_max, halfsize=True, noborder=True)
    annotateline(latticedata, ylim_max)


# annotate
start = 0.2
gap = 0.10
fs = 18
height_1 = 0.97
height_legend = height_1

# annotate active lattice
plt.annotate(str(latticedata.LatticeName), xy=(0.98, height_1), xycoords='figure fraction', va='center', ha='right', fontsize=fs - 6)
annolist_1 = 'Q$_x$: ' + str(round(twissdata.Qx, 4)) + '(' + str(round(twissdata.QxFreq, 0)) + ' kHz)   Q$_y$: ' + str(round(twissdata.Qy, 4)) + '(' + str(round(twissdata.QyFreq, 0)) + ' kHz)   $\\alpha_C$: ' + "%.3e" % twissdata.alphac
plt.annotate(annolist_1, xy=(start, height_1), xycoords='figure fraction', va='center', ha='left', fontsize=fs - 4)

# annotate beta eta
plt.annotate('$\\beta_x$/m', xy=(0.05, height_legend), color=mpl.cm.Set1(0 / 9), xycoords='figure fraction', va='center', ha='right', fontsize=fs)
plt.annotate('$\\beta_y$/m', xy=(0.1025, height_legend), color=mpl.cm.Set1(1 / 9), xycoords='figure fraction', va='center', ha='right', fontsize=fs)
plt.annotate('$\\eta_x$/10cm', xy=(0.18, height_legend), color=mpl.cm.Set1(2 / 9), xycoords='figure fraction', va='center', ha='right', fontsize=fs)

# save fig
plt.tight_layout()
plt.savefig("../../images/04-design-lattice.pdf")
