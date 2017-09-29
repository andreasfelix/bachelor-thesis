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



fig = plt.figure(figsize=(16, 9), facecolor='white')
row_length = 8
rows = 3
ylim_min = -2
ylim_max = 36
start_section = 37.5
length = 15

# reference lattice
reflatticepath = os.path.expanduser("~") + "/git/element/lattices/standard/BII_2017-08-04_23-42_LOCOFitByPS_noID_ActualUserMode.lte"
reflatticedata = returnlatticedata(reflatticepath, 'felix_full')
reftwissdata = returntwissdata(reflatticedata)

# plot
activelattice = os.path.expanduser("~") + "/git/element/lattices/Q5T2off/LOCO/BII_2017-08-05_03-09_LOCOFitByPS_noID_O5T2_off_V4.lte"
latticedata = returnlatticedata(activelattice, 'felix_full')
twissdata = returntwissdata(latticedata)

plt.subplot2grid((rows, 1), (0, 0), colspan=1, rowspan=1)

plt.plot(reftwissdata.Cs, reftwissdata.betax, '--', color=cmap(0 / 9), linewidth=1.5, label="$\\beta_x$ / m")
plt.plot(reftwissdata.Cs, reftwissdata.betay, '--', color=cmap(1 / 9), linewidth=1.5, label="$\\beta_y$ / m")
plt.plot(reftwissdata.Cs, reftwissdata.etax * 10, '--', color=cmap(2 / 9), linewidth=1.5, label="$\\eta_x$ / cm")

plt.plot(twissdata.Cs, twissdata.betax, color=cmap(0 / 9), linewidth=1.5, label="$\\beta_x$ / m")
plt.plot(twissdata.Cs, twissdata.betay, color=cmap(1 / 9), linewidth=1.5, label="$\\beta_y$ / m")
plt.plot(twissdata.Cs, twissdata.etax * 10, color=cmap(2 / 9), linewidth=1.5, label="$\\eta_x$ / cm")

plt.xlim(0, latticedata.LatticeLength)
plt.ylim(ylim_min, ylim_max)

paintlattice(0, latticedata.LatticeLength, latticedata, ylim_min, ylim_max, halfsize=True, noborder=True)
annotateline(latticedata, ylim_max)

# ticks
plt.gca().xaxis.grid(which='minor', linestyle='dotted')
plt.gca().set_xticks(np.linspace(0, latticedata.LatticeLength, 17, endpoint=True))
plt.gca().set_xticks(np.linspace(latticedata.LatticeLength / 32, latticedata.LatticeLength * (1 - 1 / 32), 16, endpoint=True), minor=True)
plt.gca().set_xlabel('orbit position $s$ / m')
plt.gca().yaxis.grid(alpha=0.5, zorder=0, linestyle='dotted')
plt.gca().set_yticks(np.linspace(0, 36, 10, endpoint=True))

# zoom
ylim_min = -2
ylim_max = 36

plt.subplot2grid((rows, 1), (1, 0), colspan=1, rowspan=2)

plt.plot(reftwissdata.Cs, reftwissdata.betax, '--', color=cmap(0 / 9), linewidth=1.5, label="$\\beta_x$ / m")
plt.plot(reftwissdata.Cs, reftwissdata.betay, '--', color=cmap(1 / 9), linewidth=1.5, label="$\\beta_y$ / m")
plt.plot(reftwissdata.Cs, reftwissdata.etax * 10, '--', color=cmap(2 / 9), linewidth=1.5, label="$\\eta_x$ / cm")

plt.plot(twissdata.Cs, twissdata.betax, color=cmap(0 / 9), linewidth=1.5, label="$\\beta_x$ / m")
plt.plot(twissdata.Cs, twissdata.betay, color=cmap(1 / 9), linewidth=1.5, label="$\\beta_y$ / m")
plt.plot(twissdata.Cs, twissdata.etax * 10, color=cmap(2 / 9), linewidth=1.5, label="$\\eta_x$ / cm")

xzmin = 27.5
xzmax = 62.5

# ticks
plt.gca().xaxis.grid(which='minor', linestyle='dotted')
plt.gca().set_xticks(np.linspace(0, latticedata.LatticeLength, 121, endpoint=True))
plt.gca().set_xticks(np.linspace(latticedata.LatticeLength / 32, latticedata.LatticeLength * (1 - 1 / 32), 16, endpoint=True), minor=True)
plt.gca().set_xlabel('orbit position $s$ / m')
plt.gca().yaxis.grid(alpha=0.75, zorder=0, linestyle='dotted')
plt.gca().set_yticks(np.linspace(0, 36, 10, endpoint=True))

paintlattice(xzmin, xzmax, latticedata, ylim_min, ylim_max, halfsize=True, noborder=True)
annotateline(latticedata, ylim_max)
plt.xlim(xzmin, xzmax)
plt.ylim(ylim_min, ylim_max)

# annotate
start = 0.225
gap = 0.10
fs = 18
offsett = 0.015
height_1 = 0.98
height_legend = height_1

# annotate active lattice
plt.annotate("V4 vs standard (LOCO)", xy=(0.98, height_1), xycoords='figure fraction', va='center', ha='right', fontsize=fs, weight='bold')
annolist_string = "$Q_x$: {:.2f} ({:.0f} kHz)   $Q_y$: {:.2f} ({:.0f} kHz)   $\\alpha_C$: {:.2e}".format(twissdata.Qx, twissdata.QxFreq, twissdata.Qy, twissdata.QyFreq, twissdata.alphac)
plt.annotate(annolist_string, xy=(start, height_1), xycoords='figure fraction', va='center', ha='left', fontsize=fs)

# annotate beta eta
plt.annotate('$\\beta_x$/m', xy=(0.0525, height_legend), color=mpl.cm.Set1(0 / 9), xycoords='figure fraction', va='center', ha='right', fontsize=fs + 2)
plt.annotate('$\\beta_y$/m', xy=(0.1075, height_legend), color=mpl.cm.Set1(1 / 9), xycoords='figure fraction', va='center', ha='right', fontsize=fs + 2)
plt.annotate('$\\eta_x$/10cm', xy=(0.1925, height_legend), color=mpl.cm.Set1(2 / 9), xycoords='figure fraction', va='center', ha='right', fontsize=fs + 2)

# save fig
plt.tight_layout()
plt.gcf().subplots_adjust(top=0.945)
plt.savefig("../../images/05-V4_vs_standard_loco.pdf")
plt.show()