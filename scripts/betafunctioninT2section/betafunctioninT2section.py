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

activelattice = expanduser("~") + "/git/element/lattices/standard/BII_2016-06-10_user_Sym_noID_DesignLattice1996.lte"
latticedata = returnlatticedata(activelattice, 'felix_normal')
twissdata = returntwissdata(latticedata)

fig = plt.figure(figsize=(8, 6), dpi=300)

center = 45
diff1 = 5.72
start = center - diff1
end = center + diff1
ylim_max = 24
ylim_min = -1

diff2 = 1.65
pos1 = center - diff2
pos2 = center + diff2

# image
extent = [start, end, ylim_min, ylim_max]
shape = (247, 932)
ax2 = plt.subplot2grid((5, 1), (3, 0), colspan=1, rowspan=2)
img = mpimg.imread('T2section_crop.png')
plt.imshow(img, extent=extent, aspect='auto')
ax2.set_yticks([])
plt.xlabel("oribit position $s$ / m")
# ax2.get_xaxis().set_tick_params(direction='in')

plt.plot([pos1, pos1], [ylim_min, ylim_max], "--r", linewidth=1.0, dashes=(5, 10))
plt.plot([pos2, pos2], [ylim_min, ylim_max], "--r", linewidth=1.0, dashes=(5, 10))

plt.xticks(np.linspace(40, 50, 11))
labels = plt.gca().get_xticks().tolist()
labels = [label - (start + end) / 2 for label in labels]
plt.gca().set_xticklabels(labels)

# graph
ax1 = plt.subplot2grid((5, 1), (0, 0), colspan=1, rowspan=3)

plt.plot(twissdata.Cs, twissdata.betax, color=cmap(0 / 9), linewidth=1.5, label="$\\beta_x$ / m")
plt.plot(twissdata.Cs, twissdata.betay, color=cmap(1 / 9), linewidth=1.5, label="$\\beta_y$ / m")
plt.plot(twissdata.Cs, twissdata.etax * 100, color=cmap(2 / 9), linewidth=1.5, label="$\\eta_x$ / cm")

plt.plot([start, end], [4, 4], "--k", linewidth=2.0, dashes=(5, 6))
plt.plot([pos1, pos1], [ylim_min, ylim_max], "--r", linewidth=1.0, dashes=(5, 10))
plt.plot([pos2, pos2], [ylim_min, ylim_max], "--r", linewidth=1.0, dashes=(5, 10))

plt.xlim(start, end)
plt.ylim(ylim_min, ylim_max)

ax1.tick_params(labelbottom='off')
ax1.grid(dashes=(5, 12), linewidth=0.5)
#
# ax1.get_yaxis().set_tick_params(direction='in')
# ax1.get_xaxis().set_tick_params(direction='inout')



ax1.set_yticks(np.linspace(0, 24, 7, endpoint=True))

plt.text(center, 2.25, "$\\beta$ = 4 m", ha='center')
plt.legend(loc="upper center")
paintlattice(start, end, latticedata, ylim_min, ylim_max, halfsize=True, noborder=True)
annotateline(latticedata, 16)

# adjust both plots
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig("../../images/04-betafunctioninT2.pdf", transparent=True)
# plt.show()
