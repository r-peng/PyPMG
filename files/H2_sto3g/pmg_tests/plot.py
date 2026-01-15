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
ax1 = ax.twinx()
def read(out):
    e0 = []
    e1 = []
    theta0 = []
    theta1 = []
    ecore = []
    for l in out:
        if l[:len('ecore')]=='ecore':
            ls = l.split(' ')
            ecore.append(float(ls[1]))
        if l[:len('E0')]=='E0':
            ls = l.split(' ')
            e0.append(float(ls[1]))
        if l[:len('E1')]=='E1':
            ls = l.split(' ')
            e1.append(float(ls[1]))
        if l[:len('theta0')]=='theta0':
            ls = l.split(' ')
            theta0.append(float(ls[1]))
        if l[:len('theta1')]=='theta1':
            ls = l.split(' ')
            theta1.append(float(ls[1]))
    ecore = np.array(ecore)
    e0 = np.array(e0)
    e1 = np.array(e1)
    theta0 = np.array(theta0)
    theta1 = np.array(theta1)
    de = e0-e1
    assert len(de[de>0])==len(de)
    return e1+ecore,theta1 

fnames = 'all_R',
colors = 'tab:blue','tab:green','tab:orange','tab:red','tab:cyan'

Rs = np.arange(1,2.01,.01)
#for f,c in zip(fnames,colors):
out = open('all_R.out', 'r').readlines()
e,theta = read(out)
ax.plot(Rs,theta,linestyle='-',color='tab:blue',label=r'$\theta$')
ax1.plot(Rs,e,linestyle='-',color='tab:orange',label=r'$E_0$')

ax.set_ylabel(r'$\theta$')
ax.set_xlabel(r'$R$')
#ax.set_yscale('log')
#ax.set_xlim(0,50)
#ax.set_ylim(-3.2,-1.8)
#ax.legend('upper left')
#ax.set_ylim(de_lim)

ax1.set_ylabel(r'$E_0$')
#ax1.legend('lower left')

fig.subplots_adjust(left=0.17, bottom=0.13, right=0.83, top=0.99)
fig.savefig(f"H2_sto3g.png", dpi=250)
plt.close(fig)
