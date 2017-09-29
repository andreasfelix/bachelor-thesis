# -*- coding: utf-8 -*-
from  __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

print('Matplotlib version: {}'.format(mpl.__version__))

from matplotlib import rc

# # for Palatino and other serif fonts use:
# rc('font', **{'family': 'serif', 'serif': ['Palatino']})
# rc('text', usetex=True)

myred = '#c1151a'

cmap = mpl.cm.Set1
cmap2 = mpl.cm.rainbow
SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize

import sys
from os.path import expanduser

sys.path.append(expanduser("~") + "/git")
from element.pyfiles.compute.transfermatrix_element_slicing import returnlatticedata
from element.pyfiles.compute.tracking import returntrackingdata
from element.pyfiles.compute.twissdata import returntwissdata
from element.pyfiles.visualize import latticeplot


class AttrClass:
    pass


def particle_distribution(N, emittance):
    X0 = np.zeros((5, N))
    phi = np.linspace(0, 2 * (1 - 1 / N) * np.pi, N).reshape(1, N)
    X0[0, :] = emittance ** 0.5 * np.sin(phi) * twissdata.betax[0] ** 0.5
    X0[1, :] = emittance ** 0.5 * np.cos(phi) / twissdata.betax[0] ** 0.5
    X0[2:5, :] = np.zeros((3, N))
    return X0


def plot(marker):
    ylim_max = 1.0  # 1.5 * 1000 * np.max(trackingdata.xtrack)
    for i in range(tracksett.rounds):
        latticeplot.paintlattice(0, latticedata.LatticeLength, latticedata, -ylim_max, ylim_max, anno=True, paintstart=latticedata.LatticeLength * i, onecolor=True, halfsize=True, noborder=False, fs=11)
        plt.plot(twissdata.Cs + latticedata.LatticeLength * i, 1000 * twissdata.betax ** 0.5 * emittance ** 0.5, marker, color='black', linewidth=2.5)
        plt.plot(twissdata.Cs + latticedata.LatticeLength * i, -1000 * twissdata.betax ** 0.5 * emittance ** 0.5, marker, color='black', linewidth=2.5)

    # annotate
    plt.xlabel('orbit position $s$ / m')
    plt.ylabel('transveral offsett $x$ / mm')
    plt.gca().set_ylim(-ylim_max, ylim_max)
    plt.gca().set_xlim(0, latticedata.LatticeLength * tracksett.rounds)

    if False:
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.xticks([])
        plt.yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
        plt.gca().add_patch(mpl.patches.FancyArrow(0, 0, 0, 0.75 * ylim_max, clip_on=False, fc='black', width=0.01, head_width=0.15, head_length=0.02))
        plt.gca().add_patch(mpl.patches.FancyArrow(0, 0, 0, -0.75 * ylim_max, clip_on=False, fc='black', width=0.01, head_width=0.15, head_length=0.02))
        plt.gca().add_patch(mpl.patches.FancyArrow(0, 0, latticedata.LatticeLength * tracksett.rounds, 0, clip_on=False, fc='black', width=0.00125, head_width=0.02, head_length=0.15))


if __name__ == '__main__':
    # track settings

    emittance = 5e-9
    tracksett = AttrClass
    tracksett.rounds = 2
    tracksett.N = 23
    tracksett.X0_width = (1e-3, 1e-3, 0, 0, 0)
    tracksett.random = False

    fig = plt.figure('Tracking', figsize=(18, 12), facecolor='white')

    # fodo min
    fig.add_subplot(221)
    props = dict(boxstyle='round', facecolor='white')
    plt.gca().text(0.05, 0.9, "1", transform=plt.gca().transAxes, fontsize=26, verticalalignment='top', bbox=props)

    plt.text(0.5, - 0.9, "$k_{QF} = 0.02$, $k_{QD} = -0.02$", color='black', fontsize=16, bbox=props)
    fodolattice_min = 'FODOCell_ohne_Dipol_min.lte'
    latticedata = returnlatticedata(fodolattice_min, 'felix_full')
    twissdata = returntwissdata(latticedata)
    tracksett.X0 = particle_distribution(tracksett.N, emittance)
    trackingdata = returntrackingdata(latticedata, tracksett)

    for i in range(tracksett.N):
        plt.plot(trackingdata.Cs, 1000 * trackingdata.xtrack[:, i], '-', color=cmap2(i / tracksett.N))
    plot('-')

    # fodo min instable
    fig.add_subplot(223)
    props = dict(boxstyle='round', facecolor='white')
    plt.gca().text(0.05, 0.9, "3", transform=plt.gca().transAxes, fontsize=26, verticalalignment='top', bbox=props)

    plt.text(0.5, - 0.9, "$k_{QF} = 0.02$, $k_{QD} = -0.07$", color='black', fontsize=16, bbox=props)
    fodolattice_min = 'FODOCell_ohne_Dipol_min_instable.lte'
    latticedata = returnlatticedata(fodolattice_min, 'felix_full')
    tracksett.X0 = particle_distribution(tracksett.N, emittance)
    trackingdata = returntrackingdata(latticedata, tracksett)

    for i in range(tracksett.N):
        plt.plot(trackingdata.Cs, 1000 * trackingdata.xtrack[:, i], '-', color=cmap2(i / tracksett.N))
    plot('--')

    # fodo_max
    fig.add_subplot(222)
    props = dict(boxstyle='round', facecolor='white')
    plt.gca().text(0.05, 0.9, "2", transform=plt.gca().transAxes, fontsize=26, verticalalignment='top', bbox=props)

    fodolattice_max = 'FODOCell_ohne_Dipol_max.lte'
    plt.text(0.5, - 0.9, "$k_{QF} = 0.2338$, $k_{QD} = -0.2338$", color='black', fontsize=16, bbox=props)
    latticedata = returnlatticedata(fodolattice_max, 'felix_full')
    twissdata = returntwissdata(latticedata)
    tracksett.X0 = particle_distribution(tracksett.N, emittance)
    trackingdata = returntrackingdata(latticedata, tracksett)

    for i in range(tracksett.N):
        plt.plot(trackingdata.Cs, 1000 * trackingdata.xtrack[:, i], '-', color=cmap2(i / tracksett.N))
    plot('-')

    # fodo max instable
    fig.add_subplot(224)
    props = dict(boxstyle='round', facecolor='white')
    plt.gca().text(0.05, 0.9, "4", transform=plt.gca().transAxes, fontsize=26, verticalalignment='top', bbox=props)

    plt.text(0.5, - 0.9, "$k_{QF} = 0.26$, $k_{QD} = -0.26$", color='black', fontsize=16, bbox=props)
    fodolattice_min = 'FODOCell_ohne_Dipol_max_instable.lte'
    latticedata = returnlatticedata(fodolattice_min, 'felix_full')
    tracksett.X0 = particle_distribution(tracksett.N, emittance)
    trackingdata = returntrackingdata(latticedata, tracksett)

    for i in range(tracksett.N):
        plt.plot(trackingdata.Cs, 1000 * trackingdata.xtrack[:, i], '-', color=cmap2(i / tracksett.N))
    plot('--')

    plt.tight_layout(pad=3)
    # plt.show()
    plt.savefig('../../images/05-limits-of-FODOcell.pdf')
