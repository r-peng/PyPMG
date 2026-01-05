import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import h5py 
from scipy import optimize
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4,4.8)})
#plt.rcParams.update({'figure.figsize':(6.4,4.8*4)})
#plt.rcParams.update({'font.family':'serif'})
#plt.rcParams.update({'text.usetex':True})
np.set_printoptions(suppress=True,precision=6)

colors = 'r','g','b','y','c','pink','orange','grey'
eref = -0.48296 # from https://doi.org/10.1103/PhysRevB.98.241109
eref = -0.48373 # from https://doi.org/10.1103/PhysRevB.98.241109
tmpdirs = {'sr_5e3':(0,'b',None,0,1),
          'sr_1e4_2.1':(0,'y',None,0,1),
          'sr_2e4':(0,'c',None,0,1),
          'rgn_5e3_2.2':(0,'b',5000,1,2),
          'rgn_1e4_2.2':(0,'y',10000,0,2),
          'rgn_2e4_2.2':(0,'c',20000,2,2),
           }

string = 'actual=(array(['
fig,ax = plt.subplots(nrows=1,ncols=1)
def read(out):
    e = []
    for l in out:
        if l[:len('step')]!='step':
            continue
        ls = l.split(',')
        e.append(float(ls[1].split('=')[-1]))
    return np.array(e)
def fit1(y):
    x = np.arange(len(y))
    logy = y-y[0]
    logy = np.log(-logy[1:])
    logx = np.log(x[1:])

    A = np.stack([logx,np.ones(len(logx))],axis=1)
    (p,a),r,_,_ = np.linalg.lstsq(A,logy,rcond=1e-6)
    a = np.exp(a)
    print('p,a,r=',p,a,r)
    return p,a,x,-a*x**p+y[0]
def fit2(y):
    x = np.arange(len(y))
    logx = np.log(x+1)
    #A = np.stack([logx,np.ones(len(logx))],axis=1)
    A = logx[1:].reshape(len(logx)-1,1)
    p,r,_,_ = np.linalg.lstsq(A,y[1:]-y[0],rcond=1e-6)
    print('p,r=',p,r)
    return p,x,p*logx+y[0]
for tmpdir,(start,color,label,cut,fit) in tmpdirs.items():
    out = open(f'{tmpdir}/out.out', 'r').readlines()
    print(tmpdir)
    e = read(out)
    x = range(start,len(e)+start)
    ls = '--' if tmpdir[:2]=='sr' else '-'
    mk = None if tmpdir[:2]=='sr' else 'o'
    ax.plot(x,e,linestyle=ls,marker=mk,markersize=3,color=color,label=label)
    ax.plot((0,50),(eref,eref),linestyle='-',color='k')
    continue
    if fit==1:
        _,_,x,y = fit1(rel_err[cut:])
    else:
        _,x,y = fit2(rel_err[cut:])
    ax.plot(x+cut,y,linestyle=':',color=color)

ax.set_ylabel('e')
ax.set_xlabel('step')
ax.set_xlim(0,50)
#ax.set_ylim(-3.2,-1.8)
ax.legend()
#ax.set_ylim(de_lim)
fig.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95)
fig.savefig(f"8x8_D4_energy.png", dpi=250)
plt.close(fig)
