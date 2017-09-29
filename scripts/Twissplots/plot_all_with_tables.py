# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

import pandas as pd
import os
import time
import sys

sys.path.append(os.path.expanduser("~") + "/git")

from element.pyfiles.compute.transfermatrix_element_slicing import returnlatticedata
from element.pyfiles.compute.twissdata import returntwissdata
from element.pyfiles.visualize.latticeplot import paintlattice, annotateline

# Reference lattice
reflatticepath = os.path.expanduser("~") + "/git/element/lattices/standard/BII_2017-03-28_17-54_LOCOFitByPS_noID_ActualUserMode_third_best.lte"
reflatticedata = returnlatticedata(reflatticepath)
reftwissdata = returntwissdata(reflatticedata, interpolate=1000)
betaxmax = np.max(reftwissdata.betax)
betaymax = np.max(reftwissdata.betay)
int_Cs = np.linspace(0, reflatticedata.LatticeLength, 1e4)
int_ref_betax = np.interp(int_Cs, reftwissdata.Cs, reftwissdata.betax)
int_ref_betay = np.interp(int_Cs, reftwissdata.Cs, reftwissdata.betay)
latticelist = [os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/V1/V1",
               os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/V2/V2_2017-05-14_15-42-32",
               os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/V3/V3_2017-05-14_15-45-23",
               os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/V4/V4_2017-05-14_15-55-33",
               os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/V5/V5_2017-05-14_05-17-07",
               os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/Vall/Vall_2017-08-01_15-35-13",
               os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/V2Q3T/V2Q3T_2017-07-31_16-00-02",
               os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/V2Q4T/V2Q4T_2017-07-31_15-31-53",
               os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/V2Q5/V2Q5_2017-07-31_15-47-44",
               os.path.expanduser("~") + "/git/element/lattices/Q5T2off/SIM/VOF/VOF_2017-08-08_18-39-04"]

names = ['V1', 'V2', 'V3', 'V4', 'V5', 'Vall', 'V2Q3T', 'V2Q4T', 'V2Q5', 'VOF']
scalarvalueslist = [{'$Q_{\\textup{x}}$ / kHz': reftwissdata.QxFreq, '$Q_{\\textup{y}}$ / kHz': reftwissdata.QyFreq, '$\\beta_{\\textup{x,max}}$ / m': betaxmax, '$\\beta_{\\textup{y,max}}$ / m': betaymax,
                     '$\overline{\\beta}_{\\textup{x,rel}}$ / m': 1, '$\overline{\\beta}_{\\textup{y,rel}}$ / m': 1}]

fig = plt.figure(figsize=(16, 9), facecolor='white')
row_length = 8
rows = 3
ylim_min = -2
ylim_max = 32
start_section = 37.5
length = 15

for i, lattice in enumerate(latticelist):
    latticedata = returnlatticedata(lattice + '.lte', 'felix_full')
    twissdata = returntwissdata(latticedata, interpolate=1000)
    betaxmax = np.max(twissdata.betax)
    betaymax = np.max(twissdata.betay)

    beta_mean_res_x = np.mean(twissdata.betax_int / reftwissdata.betax_int)
    beta_mean_res_y = np.mean(twissdata.betay_int / reftwissdata.betay_int)

    scalarvalues = {'$Q_{\\textup{x}}$ / kHz': twissdata.QxFreq, '$Q_{\\textup{y}}$ / kHz': twissdata.QyFreq, '$\\beta_{\\textup{x,max}}$ / m': betaxmax, '$\\beta_{\\textup{y,max}}$ / m': betaymax,
                    '$\overline{\\beta}_{\\textup{x,rel}}$ / m': beta_mean_res_x, '$\overline{\\beta}_{\\textup{y,rel}}$ / m': beta_mean_res_y}
    df = pd.DataFrame(data=scalarvalues, index=[0])
    df = df.round(2)

    scalarvalueslist.append(scalarvalues)
    B = pd.read_csv(lattice + '.mag', comment='#', names=["\\textbf{Magnets}", "\\textbf{Initial}", "\\textbf{Final}", "\\textbf{Difference}", "\\textbf{Factor}"])
    B.index += 1
    with open('../../tables/' + names[i] + '.tex', 'wb') as outfile:
        df.to_latex(outfile, index=False, escape=False)
        outfile.write("\\\\")
        B.to_latex(outfile, bold_rows=True, escape=False)

    # plot all
    plt.subplot2grid((rows, 1), (0, 0), colspan=1, rowspan=1)

    plt.plot(reftwissdata.Cs, reftwissdata.betax, '--', color=cmap(0 / 9), linewidth=1.5, label="$\\beta_x$ / m")
    plt.plot(reftwissdata.Cs, reftwissdata.betay, '--', color=cmap(1 / 9), linewidth=1.5, label="$\\beta_y$ / m")
    plt.plot(reftwissdata.Cs, reftwissdata.etax * 10, '--', color=cmap(2 / 9), linewidth=1.5, label="$\\eta_x$ / cm")

    plt.plot(twissdata.Cs, twissdata.betax, color=cmap(0 / 9), linewidth=1.5, label="$\\beta_x$ / m")
    plt.plot(twissdata.Cs, twissdata.betay, color=cmap(1 / 9), linewidth=1.5, label="$\\beta_y$ / m")
    plt.plot(twissdata.Cs, twissdata.etax * 10, color=cmap(2 / 9), linewidth=1.5, label="$\\eta_x$ / cm")

    plt.xlim(0, latticedata.LatticeLength)
    plt.ylim(ylim_min, ylim_max)

    # ticks
    plt.gca().xaxis.grid(which='minor', linestyle='dotted')
    plt.gca().set_xticks(np.linspace(0, latticedata.LatticeLength, 17, endpoint=True))
    plt.gca().set_xticks(np.linspace(latticedata.LatticeLength / 32, latticedata.LatticeLength * (1 - 1 / 32), 16, endpoint=True), minor=True)
    plt.gca().set_xlabel('orbit position $s$ / m')
    plt.gca().yaxis.grid(alpha=0.5, zorder=0, linestyle='dotted')

    if i == 0:
        plt.gca().set_yticks(np.linspace(0, 56, 8, endpoint=True))
        paintlattice(0, latticedata.LatticeLength, latticedata, ylim_min, 56, halfsize=True, noborder=True)
        annotateline(latticedata, 56)

    elif i == 1:
        plt.gca().set_yticks(np.linspace(0, 40, 11, endpoint=True))
        paintlattice(0, latticedata.LatticeLength, latticedata, ylim_min, 40, halfsize=True, noborder=True)
        annotateline(latticedata, 40)
    else:
        plt.gca().set_yticks(np.linspace(0, 32, 9, endpoint=True))
        paintlattice(0, latticedata.LatticeLength, latticedata, ylim_min, ylim_max, halfsize=True, noborder=True)
        annotateline(latticedata, ylim_max)

    # zoom
    plt.subplot2grid((rows, 1), (1, 0), colspan=1, rowspan=2)

    plt.plot(reftwissdata.Cs, reftwissdata.betax, '--', color=cmap(0 / 9), linewidth=1.5, label="$\\beta_x$ / m")
    plt.plot(reftwissdata.Cs, reftwissdata.betay, '--', color=cmap(1 / 9), linewidth=1.5, label="$\\beta_y$ / m")
    plt.plot(reftwissdata.Cs, reftwissdata.etax * 10, '--', color=cmap(2 / 9), linewidth=1.5, label="$\\eta_x$ / cm")

    plt.plot(twissdata.Cs, twissdata.betax, color=cmap(0 / 9), linewidth=1.5, label="$\\beta_x$ / m")
    plt.plot(twissdata.Cs, twissdata.betay, color=cmap(1 / 9), linewidth=1.5, label="$\\beta_y$ / m")
    plt.plot(twissdata.Cs, twissdata.etax * 10, color=cmap(2 / 9), linewidth=1.5, label="$\\eta_x$ / cm")



    # ticks
    plt.gca().xaxis.grid(which='minor', linestyle='dotted')
    plt.gca().set_xticks(np.linspace(0, latticedata.LatticeLength, 121, endpoint=True))
    plt.gca().set_xticks(np.linspace(latticedata.LatticeLength / 32, latticedata.LatticeLength * (1 - 1 / 32), 16, endpoint=True), minor=True)
    plt.gca().set_xlabel('orbit position $s$ / m')
    plt.gca().yaxis.grid(alpha=0.75, zorder=0, linestyle='dotted')

    xzmin = 27.5
    xzmax = 62.5

    plt.xlim(xzmin, xzmax)
    plt.ylim(ylim_min, ylim_max)

    if i == 0 or i == 1:
        plt.gca().set_yticks(np.linspace(0, 36, 10, endpoint=True))
        paintlattice(xzmin, xzmax, latticedata, ylim_min, 36, halfsize=True, noborder=True)
        annotateline(latticedata, 36)
    else:
        plt.gca().set_yticks(np.linspace(0, 32, 9, endpoint=True))
        paintlattice(xzmin, xzmax, latticedata, ylim_min, ylim_max, halfsize=True, noborder=True)
        annotateline(latticedata, ylim_max)


    # annotate
    start = 0.225
    gap = 0.10
    fs = 18
    offsett = 0.015
    height_1 = 0.98
    height_legend = height_1

    # annotate active lattice
    plt.annotate(names[i], xy=(0.98, height_1), xycoords='figure fraction', va='center', ha='right', fontsize=fs, weight='bold')
    annolist_string = "$Q_x$: {:.2f} ({:.0f} kHz)   $Q_y$: {:.2f} ({:.0f} kHz)   $\\alpha_C$: {:.2e}".format(twissdata.Qx, twissdata.QxFreq, twissdata.Qy, twissdata.QyFreq, twissdata.alphac)
    plt.annotate(annolist_string, xy=(start, height_1), xycoords='figure fraction', va='center', ha='left', fontsize=fs)

    # annotate beta eta
    plt.annotate('$\\beta_x$/m', xy=(0.0525, height_legend), color=mpl.cm.Set1(0 / 9), xycoords='figure fraction', va='center', ha='right', fontsize=fs + 2)
    plt.annotate('$\\beta_y$/m', xy=(0.1075, height_legend), color=mpl.cm.Set1(1 / 9), xycoords='figure fraction', va='center', ha='right', fontsize=fs + 2)
    plt.annotate('$\\eta_x$/10cm', xy=(0.1925, height_legend), color=mpl.cm.Set1(2 / 9), xycoords='figure fraction', va='center', ha='right', fontsize=fs + 2)

    # save fig
    plt.tight_layout(h_pad=2)
    plt.gcf().subplots_adjust(top=0.945)
    plt.savefig("../../images/Overview-all-solutions/" + names[i] + "-comparison.pdf", transparent=True)

df = pd.DataFrame(scalarvalueslist)
df = df.round(2)
df.insert(0, 'Version', ['current'] + names)
with open('../../tables/V-comparison.tex', 'wb') as outfile:
    df.to_latex(outfile, index=False, escape=False)
