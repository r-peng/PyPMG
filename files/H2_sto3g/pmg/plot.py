import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import h5py 
from scipy import optimize
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4,4.8)})
np.set_printoptions(suppress=True,precision=6)

fig,ax = plt.subplots(nrows=1,ncols=1)
def read(out):
    e = []
    for l in out:
        if l[:len('step')]!='step':
            continue
        ls = l.split(',')
        e.append(float(ls[1].split('=')[-1]))
    return np.array(e)

out = open(f'pmg.out', 'r').readlines()
e = read(out)
ax.plot(range(len(e)),e,linestyle='-',color='tab:blue',label='VMC')

eref = -1.63032754
ax.plot((0,len(e)),(eref,eref),linestyle='--',color='k')

ax.set_ylabel('E')
ax.set_xlabel('step')
#ax.set_xlim(0,50)
#ax.set_ylim(-3.2,-1.8)
ax.legend()
#ax.set_ylim(de_lim)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.99, top=0.99)
fig.savefig(f"H2_sto3g.png", dpi=250)
plt.close(fig)
