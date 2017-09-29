# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg

print('Matplotlib version: {}'.format(mpl.__version__))
SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize

cmap = mpl.cm.Set1
from matplotlib import rc
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot

# # for Palatino and other serif fonts use:
# rc('font', **{'family': 'serif', 'serif': ['Palatino']})
# rc('text', usetex=True)

filepath = "20170517_Q5offT2_AllVersions.str"

data = np.loadtxt(filepath, usecols=(3, 4), skiprows=1)

efficiency = data[:, 0]
current = data[:, 1]

# change values over 100 % to last value
for i, value in enumerate(efficiency):
    if value > 100:
        efficiency[i] = lastvalue
    else:
        lastvalue = value

# mask data
start = 250
end = len(efficiency)
efficiency = efficiency[start:end]
current = current[start:end]

fig = plt.figure(facecolor='white', figsize=(21, 9))
ax = host_subplot(111, axes_class=AA.Axes)
plt.plot(efficiency, lw=2, label='Injection efficiency in %', color=cmap(1 / 9))
plt.gca().set_ylabel('Injection efficiency in %')

startarr = np.array([410, 900, 1365, 1740, 2650, 3000, 3320, len(efficiency) + start])
startarr -= start
midpoints = (startarr[1:] + startarr[:-1]) / 2

names = ['V1', 'V2', 'V3', 'V4', 'V5', 'Vall', 'V4']
# for value in startarr:
#     plt.plot([value, value], [-100, 200], '--k')

for i in range(len(midpoints)):
    plt.text(midpoints[i], 103, names[i], horizontalalignment='center', fontsize=18, weight='bold')

meanstartarr = np.array([325, 828, 1183, 1552, 2423, 2834, 3127])
meanendarr = np.array([536, 1008, 1400, 1853, 2607, 3014, 3330])

for i, name in enumerate(names):
    mean = np.mean(efficiency[meanstartarr[i]:meanendarr[i]])
    plt.plot([meanstartarr[i], meanendarr[i]], [mean, mean], '--', color=cmap(0 / 9), lw=6)
    print('{} & {:.1f}\\,\\% \\\\'.format(name, mean))

# plt.grid()
ax.set_xticks(startarr)
ax.set_xticks(midpoints, minor='True')
ax.set_yticks(np.linspace(0, 100, 11))

ax.xaxis.grid(which='minor', linestyle='dotted')
ax.yaxis.grid(alpha=0.5, zorder=0, linestyle='dotted')

# plt.plot([0, len(current)], [100, 100], '--k')
ax2 = ax.twinx()
ax2.plot(current, lw=2, label='Storage Ring current / mA', color=cmap(4 / 9))
ax2.set_ylabel('Storage Ring current / mA')
ax2.set_ylim((-5, 300))

ax.set_xlabel('Time / s')
ax.set_xlim((0, len(current)))
ax.set_ylim((-5, 110))

ax.legend(loc='lower center')
ax.grid(True, linestyle='dashed')

plt.tight_layout()
plt.savefig('../../images/05-injection_efficiency_all_version.pdf')

ax.set_xlim(0, 2400)
plt.tight_layout()
plt.savefig('../../images/05-injection_efficiency_all_version_def.pdf')
