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

eref = -1.6303275411526
fnames = 'exact','n10000','n5000','n2000','n1000'
colors = 'tab:blue','tab:green','tab:orange','tab:red','tab:cyan'
for f,c in zip(fnames,colors):
    out = open(f+f'.out', 'r').readlines()
    y = np.fabs(read(out)-eref)
    ax.plot(range(len(y)),y,linestyle='-',color=c,label=f)

ax.set_ylabel(rf'$|\Delta E|$')
ax.set_xlabel('step')
ax.set_yscale('log')
#ax.set_xlim(0,50)
#ax.set_ylim(-3.2,-1.8)
ax.legend()
#ax.set_ylim(de_lim)
fig.subplots_adjust(left=0.16, bottom=0.15, right=0.99, top=0.99)
fig.savefig(f"H2_sto3g.png", dpi=250)
plt.close(fig)
