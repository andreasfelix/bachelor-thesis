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

SMALL_SIZE = 16
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
    X0[2, :] = emittance ** 0.5 * np.sin(phi) * twissdata.betay[0] ** 0.5
    X0[3, :] = emittance ** 0.5 * np.cos(phi) / twissdata.betay[0] ** 0.5
    X0[4, :] = np.zeros((1, N))
    return X0


pos1 = 37
pos2 = 53


def plot(round, marker):
    ylim_max = 0.5  # 1.5 * 1000 * np.max(trackingdata.xtrack)
    for i in range(tracksett.rounds):
        latticeplot.paintlattice(pos1, pos2, latticedata, -ylim_max, ylim_max, anno=False, paintstart=latticedata.LatticeLength * i, halfsize=True, noborder=False, fs=11)
        plt.plot(twissdata.Cs + latticedata.LatticeLength * i, 1000 * twissdata.betay ** 0.5 * emittance ** 0.5, marker, color='black', linewidth=2.5)
        plt.plot(twissdata.Cs + latticedata.LatticeLength * i, -1000 * twissdata.betay ** 0.5 * emittance ** 0.5, marker, color='black', linewidth=2.5)

    # annotate
    plt.xlabel('orbit position $s$ / m')
    plt.ylabel('transveral offsett $y$ / mm')
    plt.gca().set_ylim(-ylim_max, ylim_max)
    plt.gca().set_xlim(pos1 + round * latticedata.LatticeLength, pos2 + round * latticedata.LatticeLength)
    plt.gca().set_xticks(np.linspace(pos1 + 0.5 + round * latticedata.LatticeLength, pos2 - 0.5 + round * latticedata.LatticeLength, 7, endpoint=True))

    plt.gca().xaxis.grid(which='minor', linestyle='dashed')
    plt.gca().set_xticks([pos1 + 0.5 + round * latticedata.LatticeLength, pos2 - 0.5 + round * latticedata.LatticeLength], minor=True)

    plt.annotate("T2", xy=(45 + round * latticedata.LatticeLength, 0.4), fontsize=20, va='center', ha='center', clip_on=True, zorder=102)


if __name__ == '__main__':
    # track settings

    emittance = 5e-9
    tracksett = AttrClass
    tracksett.rounds = 3
    tracksett.N = 15
    tracksett.X0_width = (1e-3, 1e-3, 0, 0, 0)
    tracksett.random = False

    fig = plt.figure('Tracking', figsize=(16, 12), facecolor='white')

    # bessy 2 stable
    fig.add_subplot(321)
    props = dict(boxstyle='round', facecolor='white')
    plt.gca().text(0.05, 0.9, "turn 1", transform=plt.gca().transAxes, fontsize=20, verticalalignment='top', bbox=props)

    plt.text(0.5, - 0.9, "$k_{QF} = 0.02$, $k_{QD} = -0.02$", color='black', fontsize=16, bbox=props)
    fodolattice_min = 'BII_2017-03-28_17-54_LOCOFitByPS_noID_ActualUserMode_third_best.lte'
    latticedata = returnlatticedata(fodolattice_min, 'felix_full')
    twissdata = returntwissdata(latticedata)
    tracksett.X0 = particle_distribution(tracksett.N, emittance)
    trackingdata = returntrackingdata(latticedata, tracksett)

    for i in range(tracksett.N):
        plt.plot(trackingdata.Cs, 1000 * trackingdata.ytrack[:, i], '-', color=cmap2(i / tracksett.N))
    plot(0, '-')

    # round 2
    fig.add_subplot(323)
    plt.gca().text(0.05, 0.9, "turn 2", transform=plt.gca().transAxes, fontsize=20, verticalalignment='top', bbox=props)
    for i in range(tracksett.N):
        plt.plot(trackingdata.Cs, 1000 * trackingdata.ytrack[:, i], '-', color=cmap2(i / tracksett.N))
    plot(1, '-')

    # round 3
    fig.add_subplot(325)
    plt.gca().text(0.05, 0.9, "turn 3", transform=plt.gca().transAxes, fontsize=20, verticalalignment='top', bbox=props)
    for i in range(tracksett.N):
        plt.plot(trackingdata.Cs, 1000 * trackingdata.ytrack[:, i], '-', color=cmap2(i / tracksett.N))
    plot(2, '-')

    # bessy instable
    fig.add_subplot(322)
    props = dict(boxstyle='round', facecolor='white')
    plt.gca().text(0.05, 0.9, "turn 1", transform=plt.gca().transAxes, fontsize=20, verticalalignment='top', bbox=props)

    plt.text(0.5, - 0.9, "$k_{QF} = 0.02$, $k_{QD} = -0.07$", color='black', fontsize=16, bbox=props)
    fodolattice_min = 'BII_2017-03-28_17-54_LOCOFitByPS_noID_ActualUserMode_third_best_not_stable.lte'
    latticedata = returnlatticedata(fodolattice_min, 'felix_full')
    tracksett.X0 = particle_distribution(tracksett.N, emittance)
    trackingdata = returntrackingdata(latticedata, tracksett)
    returntwissdata(latticedata)

    for i in range(tracksett.N):
        plt.plot(trackingdata.Cs, 1000 * trackingdata.ytrack[:, i], '-', color=cmap2(i / tracksett.N))
    plot(0, '--')

    # round 2
    fig.add_subplot(324)
    plt.gca().text(0.05, 0.9, "turn 2", transform=plt.gca().transAxes, fontsize=20, verticalalignment='top', bbox=props)
    for i in range(tracksett.N):
        plt.plot(trackingdata.Cs, 1000 * trackingdata.ytrack[:, i], '-', color=cmap2(i / tracksett.N))
    plot(1, '--')

    # round 2
    fig.add_subplot(326)
    plt.gca().text(0.05, 0.9, "turn 3", transform=plt.gca().transAxes, fontsize=20, verticalalignment='top', bbox=props)
    for i in range(tracksett.N):
        plt.plot(trackingdata.Cs, 1000 * trackingdata.ytrack[:, i], '-', color=cmap2(i / tracksett.N))
    plot(2, '--')

    plt.tight_layout()
    plt.savefig("../../images/05-bessy2-stability-Q3T2.pdf")
    # plt.show()
