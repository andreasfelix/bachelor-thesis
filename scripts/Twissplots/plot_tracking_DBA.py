# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

print('Matplotlib version: {}'.format(mpl.__version__))

from matplotlib import rc

# # for Palatino and other serif fonts use:
# rc('font', **{'family': 'serif', 'serif': ['Palatino']})
# rc('text', usetex=True)

SMALL_SIZE = 18
MEDIUM_SIZE = 18
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize

cmap = mpl.cm.Set1

from element.pyfiles.newtracking.newcode import tracking
import sys
import scipy
from os.path import expanduser

sys.path.append(expanduser("~") + "/git")
from element.pyfiles.compute.latticedata import returnlatticedata
from element.pyfiles.compute.transfermatrix_element_slicing import returnlatticedata as returnlatticedata_linear
from element.pyfiles.compute.tracking import returntrackingdata
from element.pyfiles.compute.twissdata import returntwissdata
from element.pyfiles.visualize.latticeplot import paintlattice
from element.pyfiles.miscellaneous.LatticeTools import LatticeEditor

cmap2 = mpl.cm.rainbow
props = dict(boxstyle='round', facecolor='white')


class AttrClass:
    pass


def plot(XY, anno=True):
    plt.gca().set_xlim(0, latticedata.LatticeLength)
    plt.gca().set_ylim(-1e3 * ylim, 1e3 * ylim)
    paintlattice(0, latticedata.LatticeLength, latticedata, -1e3 * ylim, +1e3 * ylim, noborder=True, halfsize=True, fs=15)
    plt.xlabel('orbit postition $s$ / m')
    plt.xticks(np.linspace(0, 15, 7))
    if XY == 'X':
        for i in range(N):
            plt.plot(svec, 1e3 * xvec[:, i], color=cmap2(i / N), label='$\\delta$ = {:.1e}'.format(X_int[4, i]))
            plt.ylabel('$x$ / mm')
    else:
        for i in range(N):
            plt.plot(svec, 1e3 * yvec[:, i], color=cmap2(i / N), label='$\\delta$ = {:.1e}'.format(X_int[4, i]))
            plt.ylabel('$y$ / mm')
    if anno:
        plt.gca().text(0.05, 0.2, "K: {}, L: {}".format(K, L), transform=plt.gca().transAxes, fontsize=15, verticalalignment='top', bbox=props)


# setup
h = 0.01
L = 3.0
width = 0.0002
N = 7
ylim = 10.0 * width

# DBA
plt.figure('DBA', figsize=(16, 16), facecolor='white'),
activelattice = expanduser("~") + '/git/element/lattices/misc/Newtracking/DBA_original.lte'
latticedata = returnlatticedata(activelattice)

X_int = np.zeros((5, N))
for i in range(N):
    X_int[0, i] = 2 * (i / (N - 1) - 0.5) * 0
    X_int[2, i] = 2 * (i / (N - 1) - 0.5) * 0
    X_int[4, i] = 2 * (i / (N - 1) - 0.5) * -0.002


svec, xvec, yvec = tracking(latticedata, X_int, h_default=h)
plt.gcf().add_subplot(3, 1, 1)
plot('X', False)
plt.legend(loc='lower right', fontsize=16)

X_int = np.zeros((5, N))
for i in range(N):
    X_int[0, i] = 2 * (i / (N - 1) - 0.5) * width
    X_int[2, i] = 2 * (i / (N - 1) - 0.5) * width
    X_int[4, i] = 2 * (i / (N - 1) - 0.5) * -0.002

svec, xvec, yvec = tracking(latticedata, X_int, h_default=h)
plt.gcf().add_subplot(3, 1, 2)
plot('X', False)
plt.legend(loc='lower right', fontsize=16)

# plot twiss
plt.gcf().add_subplot(3, 1, 3)
ylim_min = -2
ylim_max = 24

activelattice2 = expanduser("~") + "/git/element/lattices/standard/BII_2016-06-10_user_Sym_noID_DesignLattice1996.lte"
latticedata_linear = returnlatticedata_linear(activelattice2, 'felix_full')
twissdata = returntwissdata(latticedata_linear)

plt.plot(twissdata.Cs, twissdata.betax, color=cmap(0 / 9), linewidth=1.5, label="$\\beta_x$ / m")
plt.plot(twissdata.Cs, twissdata.betay, color=cmap(1 / 9), linewidth=1.5, label="$\\beta_y$ / m")
plt.plot(twissdata.Cs, twissdata.etax * 10, color=cmap(2 / 9), linewidth=1.5, label="$\\eta_x$ / cm")

plt.ylabel("value")
plt.legend(loc='lower right', fontsize=16)

# ticks
plt.gca().get_xaxis().set_tick_params(direction='in')
plt.gca().get_yaxis().set_tick_params(direction='in')
# plt.gca().xaxis.grid(which='minor', linestyle='dashed')
plt.gca().set_xticks(np.linspace(latticedata_linear.LatticeLength / 32, latticedata_linear.LatticeLength * (1 - 1 / 32), 16, endpoint=True), minor=True)
plt.gca().set_xticks(np.linspace(15, 30, 7, endpoint=True))
# plt.gca().yaxis.grid(linestyle='dashed')
plt.gca().set_xlabel('orbit position $s$ / m')
plt.gca().set_yticks(np.linspace(0, 24, 7, endpoint=True))

# limits
plt.xlim(15, 30)
plt.ylim(ylim_min, ylim_max)
paintlattice(15, 30, latticedata_linear, ylim_min, ylim_max, noborder=True, halfsize=True, fs=15)

labels = plt.gca().get_xticks().tolist()
labels = [label - 15 for label in labels]
plt.gca().set_xticklabels(labels)

plt.tight_layout(h_pad=2)
plt.savefig("../../images/04-design-DBA.pdf")
