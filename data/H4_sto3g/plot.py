import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import h5py 
from scipy import optimize
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4,4.8)})
np.set_printoptions(suppress=True,precision=6)

eref = []
string = 'FCI electronic energy='
out = open(f'pyscf.out', 'r').readlines()
for l in out:
    if l[:len(string)]!=string:
        continue
    ls = l.split(' ')
    e = float(ls[-1])
    eref.append(e)
eref = np.array(eref)

string = 'E='
def read(out):
    e = 0
    for l in out:
        if l[:len(string)]!=string:
            continue
        ls = l.split(',')
        e = min(e,float(ls[0].split('=')[-1]))
    return e

fig,ax = plt.subplots(nrows=1,ncols=1)
def _plot1(fname,Rmax,linestyle,marker,color,label=None):
    Rs = np.arange(1.01,Rmax+.01,0.02)
    es = []
    for R in Rs:
        out = open(f'{fname}/R{R:.2f}/out.out', 'r').readlines()
        es.append(read(out))
    es = np.array(es)
    ax.plot(Rs,es-eref[:len(es)],linestyle=linestyle,marker=marker,color=color,label=label)

def _plot2(fname,path,R0,linestyle,marker,color,label=None):
    es = []
    string = 'E='
    out = open(f'{fname}/path{path}/out.out', 'r').readlines()
    for l in out:
        if l[:len(string)]!=string:
            continue
        ls = l.split(',')
        e = float(ls[0].split('=')[-1])
        es.append(e)
    es = np.array(es)
    start = int((R0-1.01)/0.02+1e-6)
    stop = start+len(es)
    Rs = np.arange(len(es))*0.02+R0
    ax.plot(Rs,es-eref[start:stop],linestyle=linestyle,marker=marker,color=color,label=label)

_plot1('lowdin_jastrow',1.49,'-','o','tab:blue',label='J+GHF')
_plot1('lowdin_pmg2',1.49,'-','o','tab:orange',label='PMG2+GHF')
_plot2('lowdin_pmg2',0,1.01,'--',None,'tab:orange')
_plot2('lowdin_pmg2',1,1.17,':',None,'tab:orange')
_plot2('lowdin_pmg3',0,1.01,'--',None,'tab:green')
_plot2('lowdin_pmg3',1,1.07,'--',None,'tab:green')
_plot2('lowdin_pmg3',2,1.17,'--',None,'tab:green')
_plot2('lowdin_pmg3',3,1.19,'--',None,'tab:green')
_plot1('lowdin_pmg3',1.35,'-','o','tab:green',label='PMG3+GHF')

ax.set_ylabel(rf'$\Delta E$')
ax.set_xlabel(r'R')
ax.set_yscale('log')
#ax.set_xlim(0,50)
ax.legend()
#ax.set_ylim(de_lim)

#ax1.set_ylabel(r'$\langle S^2\rangle$')
#ax1.set_yscale('log')

fig.subplots_adjust(left=0.2, bottom=0.15, right=0.99, top=0.99)
fig.savefig(f"H4_lowdin_jastrow.png", dpi=250)
plt.close(fig)
