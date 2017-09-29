# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg

print('Matplotlib version: {}'.format(mpl.__version__))

cmap = mpl.cm.Set1
SIZE_1 = 12
SIZE_2 = 16
SIZE_3 = 12

plt.rc('font', size=SIZE_1)  # controls default text sizes
plt.rc('axes', titlesize=SIZE_1)  # fontsize of the axes title
plt.rc('xtick', labelsize=SIZE_1)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE_1)  # fontsize of the tick labels
plt.rc('axes', labelsize=SIZE_2)  # fontsize of the x and y labels
plt.rc('legend', fontsize=SIZE_3)  # legend fontsize

# spacing
spacing = np.array([677, 740, 130, 678, 130, 678, 130, 740, 677]) / 1000
print(np.array([0]).shape, np.cumsum(spacing).shape)
positions = np.append(np.array([0]), np.cumsum(spacing))
positions = positions - positions[-1] / 2

start = positions[0]
end = positions[-1]
print(end)
ylim_min = 0
ylim_max = 6

fig = plt.figure(figsize=(7.2, 6), dpi=300)

# image
extent = [start, end, ylim_min, ylim_max]
ax2 = plt.subplot2grid((5, 1), (3, 0), colspan=1, rowspan=2)
img = mpimg.imread('cavity.png')
plt.imshow(img, extent=extent, aspect='auto')
ax2.set_yticks([])
plt.xticks(positions)
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.grid()
plt.xlabel("oribit position $s$ / m")

# graph
Cs = np.linspace(start, end)
betamin = np.array([0.63, 3.37])
betamin_v, Cs_v = np.meshgrid(betamin, Cs)
beta = betamin_v + Cs_v ** 2 / betamin_v

ax1 = plt.subplot2grid((5, 1), (0, 0), colspan=1, rowspan=3)
lineObjects = plt.plot(Cs, beta)
plt.legend(iter(lineObjects), ('min: 0.63 m', 'max: 3.37 m'))

plt.xlim(start, end)
plt.ylim(ylim_min, ylim_max)

ax1.tick_params(labelbottom='off')
ax1.grid(dashes=(5, 12), linewidth=0.5)

ax1.set_xticks(positions)
# ax1.set_yticks(np.linspace(0, 24, 7, endpoint=True))
plt.ylabel('$\\beta$ / m')
# adjust both plots
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig("../../images/04-betafunction_in_cavity_def.pdf", transparent=True)
# plt.show()
