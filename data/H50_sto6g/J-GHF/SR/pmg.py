import numpy as np
import scipy.linalg
import pickle,h5py,itertools
from PyPMG.pmg import * 
from PyPMG.jastrow import * 
from PyPMG.vmc import *
np.set_printoptions(precision=5,suppress=True,threshold=100000)

propose_by = 'uniform'
rho_swap = 0.
thresh = 1e-28
run = 0 
every = 50
nsample = 500 
if RANK==0:
    print('every=',every)

U0 = True
start,stop = 0,100
optimizer = 'SR'
rate1 = 0.1
rate2 = 1. 
#eigen_thresh = 1e-3
eigen_thresh = None
penalty = False 
symmetry = 'u11'
Jshift = 0.03

nsites = 50,50
nelec = 25,25
nsite = sum(nsites)
R = 2.0
jas = Jastrow(nsite=nsite,Jmax=None)
if U0:
    U0 = np.zeros((100,100))
    mo = np.load(f'../../lowdin_hfmo/r{R:.2f}.npy')
    U0[::2,::2] = mo[0]
    U0[1::2,1::2] = mo[1]
    eps = 0.1
else:
    U0 = None
    eps = 0.5
pmg = PMGState_autodiff(nsites,nelec,U0=U0,symmetry=symmetry,rho_swap=rho_swap,propose_by=propose_by,thresh=thresh)
pmg.add_pmg('GHF',None)
#pmg = PMGState_manual(nsites,nelec,U0=U0,symmetry=symmetry,rho_swap=rho_swap,propose_by=propose_by)
#pmg.add_pmg('GHF',None,jac_by='frechet')

psi = JastrowPMGState(jas,pmg)
#if start==0:
#    x = (np.random.rand(psi.nparam)*2-1)*eps
#    x[:jas.nparam] += Jshift
#    COMM.Bcast(x,root=0)
#    if RANK==0:
#        np.save(f'run{run}_start{start}.npy',x)
x = np.load(f'run{run}_start{start}.npy')
psi.update(x)

ham = dict()
f = h5py.File(f'../../lowdin/r{R:.2f}.h5','r')
const = f['ecore'][()]
eri = f['eri_oao'][:] 
hcore = f['hcore_oao'][:]
f.close()

ham['energy'] = QCHamiltonian(hcore,eri,save_link=False)
if penalty:
    ham['S^2'] = TotalSpin(hcore.shape[0],weight=0.1)

#sampler = DenseSampler(exact=True)
sampler = MHSampler(every=every)
cf = np.load(f'run{run}_start{start}_configs.npy',allow_pickle=True)
sampler.cf = int(cf[RANK%cf.shape[0]]) 

if optimizer=='SGD':
    vmc = SGD(psi,ham,sampler)
elif optimizer=='SR':
    vmc = SR(psi,ham,sampler)
elif optimizer=='RGN':
    vmc = RGN(psi,ham,sampler)
else:
    raise ValueError
vmc.eigen_thresh = eigen_thresh 
vmc.rate1 = rate1
vmc.rate2 = rate2
vmc.cond1 = 1e-3
vmc.cond2 = 1e-3
vmc.sample_size = nsample 
#fname = None
fname = f'run{run}_start'
vmc.run(start,stop,fname=fname,save_every=10)
if RANK>0:
    exit()
psi = vmc.psi
x = psi.get_x()
np.save(f'run{run}_start{stop}.npy',x)

