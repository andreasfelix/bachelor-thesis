# -*- coding: utf-8 -*-
from  __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

print('Matplotlib version: {}'.format(mpl.__version__))

# from matplotlib import rc
#
# # for Palatino and other serif fonts use:
# rc('font', **{'family': 'serif', 'serif': ['Palatino']})
# rc('text', usetex=True)

myred = '#c1151a'

cmap = mpl.cm.spectral

SIZE_1 = 14
SIZE_2 = 16
SIZE_3 = 12

plt.rc('font', size=SIZE_1)  # controls default text sizes
plt.rc('axes', titlesize=SIZE_1)  # fontsize of the axes title
plt.rc('xtick', labelsize=SIZE_1)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE_1)  # fontsize of the tick labels
plt.rc('axes', labelsize=SIZE_2)  # fontsize of the x and y labels
plt.rc('legend', fontsize=SIZE_3)  # legend fontsize

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
    X0[0, :] = emittance ** 0.5 * twissdata.betax[0] ** 0.5 * np.sin(phi)
    X0[1, :] = emittance ** 0.5 / twissdata.betax[0] ** 0.5 * np.cos(phi)
    X0[2:5, :] = np.zeros((3, N))
    return X0


if __name__ == '__main__':
    activelattice = expanduser("~") + '/git/element/lattices/misc/fodo-Cell/FODOCell_Wille_no_edge-focussing.lte'
    latticedata = returnlatticedata(activelattice, 'felix_full')

    twissdata = returntwissdata(latticedata)

    emittance = 5e-9
    tracksett = AttrClass
    tracksett.rounds = 2
    tracksett.N = 33
    tracksett.X0_width = (1e-3, 1e-3, 0, 0, 0)
    tracksett.X0 = particle_distribution(tracksett.N, emittance)
    tracksett.random = False
    trackingdata = returntrackingdata(latticedata, tracksett)


    def plot():
        ylim_max = 1.5 * 1000 * np.max(trackingdata.xtrack)
        for i in range(tracksett.rounds):
            latticeplot.paintlattice(0, latticedata.LatticeLength, latticedata, -ylim_max, ylim_max, anno=True, paintstart=latticedata.LatticeLength * i, onecolor=False, halfsize=True, fs=10)
            plt.plot(twissdata.Cs + latticedata.LatticeLength * i, 1000 * twissdata.betax ** 0.5 * emittance ** 0.5, '-', color='black', linewidth=2.5)
            plt.plot(twissdata.Cs + latticedata.LatticeLength * i, -1000 * twissdata.betax ** 0.5 * emittance ** 0.5, '-', color='black', linewidth=2.5)

        # annotate
        idx = np.searchsorted(latticedata.Cs, 1.5)
        xpos = latticedata.Cs[idx]
        ypos = 1000 * twissdata.betax[idx] ** 0.5 * emittance ** 0.5
        plt.gca().annotate('$E(s) = \\pm \\sqrt{\\epsilon \\beta(s)}$', xy=(xpos, ypos), xytext=(xpos + 0.7, ypos + 0.07), arrowprops=dict(facecolor='black', shrink=0.05, width=0.75), fontsize=16)
        # plt.gca().grid(linestyle='--', linewidth=0.05, alpha=0.3)
        plt.xlabel('orbit position $s$ / m')
        plt.ylabel('transveral offsett $u$ / mm')
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


    fig = plt.figure('Tracking', figsize=(16, 5), facecolor='1', frameon=False)

    ax = fig.add_subplot(121)
    plt.plot(trackingdata.Cs, 1000 * trackingdata.xtrack[:, 0], '--k')  # , color=cmap(i / tracksett.N) )
    # plt.plot([0, latticedata.LatticeLength * tracksett.rounds], [0, 0], '--k')
    plot()
    idx = np.searchsorted(latticedata.Cs, 1)
    xpos = trackingdata.Cs[idx]
    ypos = 1000 * trackingdata.xtrack[idx, 0]
    print(xpos, ypos)
    plt.gca().annotate('$u_i(s) = \\sqrt{\\epsilon_i} \sqrt{\\beta(s)} \\cos(\\psi(s)+\\psi_{0,i})$', xy=(xpos, ypos), xytext=(xpos + 1, ypos - 0.06), arrowprops=dict(facecolor='black', shrink=0.05, width=0.75), fontsize=16)

    ax = fig.add_subplot(122)
    for i in range(tracksett.N):
        plt.plot(trackingdata.Cs, 1000 * trackingdata.xtrack[:, i], '--k')  # , color=cmap(i / tracksett.N) )
    # plt.plot([0, latticedata.LatticeLength * tracksett.rounds], [0, 0], '--k')
    plot()

    plt.tight_layout(pad=1.5)
    plt.savefig('../images/03-envelope.pdf')
