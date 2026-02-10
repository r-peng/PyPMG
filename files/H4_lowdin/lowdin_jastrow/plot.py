import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import h5py 
from scipy import optimize
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4,4.8)})
np.set_printoptions(suppress=True,precision=6)

def read(out):
    e = []
    loss = []
    return np.array(e),np.array(loss)
Rs = np.arange(1.01,1.5,0.02)
es = []
string = 'E='
for R in Rs:
    e = 0
    out = open(f'R{R:.2f}/out.out', 'r').readlines()
    for l in out:
        if l[:len(string)]!=string:
            continue
        ls = l.split(',')
        e = min(e,float(ls[0].split('=')[-1]))
    es.append(e)
es = np.array(es)

eref = []
string = 'FCI electronic energy='
out = open(f'../pyscf.out', 'r').readlines()
for l in out:
    if l[:len(string)]!=string:
        continue
    ls = l.split(' ')
    e = float(ls[-1])
    eref.append(e)
eref = np.array(eref[:len(Rs)])
fig,ax = plt.subplots(nrows=1,ncols=1)
ax.plot(Rs,es-eref,linestyle='-',marker='o',color='tab:blue')

ax.set_ylabel(rf'$\Delta E$')
ax.set_xlabel(r'R')
#ax.set_yscale('log')
#ax.set_xlim(0,50)
#ax.set_ylim(-3.2,-1.8)
#ax.legend()
#ax.set_ylim(de_lim)

#ax1.set_ylabel(r'$\langle S^2\rangle$')
#ax1.set_yscale('log')

fig.subplots_adjust(left=0.2, bottom=0.15, right=0.99, top=0.99)
fig.savefig(f"H4_lowdin_jastrow.png", dpi=250)
plt.close(fig)
