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
#ax1 = ax.twinx()
def read(out):
    e = []
    loss = []
    for l in out:
        if l[:len('energy')]=='energy':
            ls = l.split(',')
            e.append(float(ls[0].split('=')[-1]))
        if l[:len('loss')]=='loss':
            ls = l.split(',')
            loss.append(float(ls[0].split('=')[-1]))
    return np.array(e),np.array(loss)

eref = -1.6303275411526
fnames = 'rho_0','rho_0.05','rho_0.1'
labels = r'$\rho=0$',r'$\rho=0.05$',r'$\rho=0.1$',r'$\rho=0.1$'+'test'
weights = 0.1,0.1,0.5,None
colors = 'tab:blue','tab:green','tab:orange','tab:red','tab:cyan'
for f,c,lab,w in zip(fnames,colors,labels,weights):
    out = open(f+f'.out', 'r').readlines()
    e,l = read(out)
    y = np.fabs(e-eref)
    ax.plot(range(len(y)),y,linestyle='-',color=c,label=lab)
    #y = (l-e)/w
    #print(y)
    #ax1.plot(range(len(y)),y,linestyle=':',color=c)

out = open(f'ham0_100.out', 'r').readlines()
e1,l = read(out)
out = open(f'ham100_400.out', 'r').readlines()
e2,l = read(out)
e = np.concatenate([e1,e2])
y = np.fabs(e-eref)
ax.plot(range(len(y)),y,linestyle='-',color='tab:cyan',label='ham')

ax.set_ylabel(rf'$|\Delta E|$')
ax.set_xlabel('step')
ax.set_yscale('log')
#ax.set_xlim(0,50)
#ax.set_ylim(-3.2,-1.8)
ax.legend()
#ax.set_ylim(de_lim)

#ax1.set_ylabel(r'$\langle S^2\rangle$')
#ax1.set_yscale('log')

fig.subplots_adjust(left=0.16, bottom=0.15, right=0.83, top=0.99)
fig.savefig(f"H2_sto3g_R1.0.png", dpi=250)
plt.close(fig)
