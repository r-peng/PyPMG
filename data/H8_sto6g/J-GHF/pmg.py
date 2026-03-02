import numpy as np
import scipy.linalg
import pickle,h5py,itertools
from PyPMG.pmg import * 
from PyPMG.jastrow import * 
from PyPMG.vmc import *
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(precision=10,suppress=True)

propose_by = 'uniform'
rho_swap = 0.
run = 4 
U0 = True
start,stop = 0,50
optimizer = 'RGN'
rate1 = 0.1
rate2 = 1. 
eigen_thresh = 1e-3
#eigen_thresh = None
penalty = False 
symmetry = 'u11'

#R = 1.4
#R = 1.8
R = 2.4

nsites = 8,8
nelec = 4,4
nsite = sum(nsites)
jas = Jastrow(nsite=nsite,Jmax=None)
if U0:
    U0 = np.zeros((16,16))
    mo = np.load(f'../lowdin_hfmo/r{R:.2f}.npy')
    U0[::2,::2] = mo[0]
    U0[1::2,1::2] = mo[1]
    eps = 0.1
else:
    U0 = None
    eps = 0.5
#pmg = PMGState_autodiff(nsites,nelec,U0=U0,symmetry=symmetry,rho_swap=rho_swap,propose_by=propose_by)
#pmg.add_mg('GHF')
pmg = PMGState_manual(nsites,nelec,U0=U0,symmetry=symmetry,rho_swap=rho_swap,propose_by=propose_by)
pmg.add_mg('GHF',jac_by='frechet')
#pmg.add_mg('GHF',jac_by='ad')

psi = JastrowPMGState(jas,pmg)
if start==0:
    x = (np.random.rand(psi.nparam)*2-1)*eps
    COMM.Bcast(x,root=0)
    if RANK==0:
        np.save(f'R{R:.2f}/run{run}_start{start}.npy',x)
    #x = np.load(f'R{R:.2f}/run{run}_start{start}.npy')
else:
    x = np.load(f'R{R:.2f}/run{run}_start{start}.npy')
psi.update(x)

ham = dict()
f = h5py.File(f'../lowdin/h4_{R:.2f}.h5','r')
const = f['ecore'][()]
eri = f['eri_oao'][:] 
hcore = f['hcore_oao'][:]
f.close()

ham['energy'] = QCHamiltonian(hcore,eri)
if penalty:
    ham['S^2'] = TotalSpin(hcore.shape[0],weight=0.1)

sampler = DenseSampler(exact=True)
#sampler = MHSampler(burn_in=20,every=20)
sampler.cf = (1,1,0,0)
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
vmc.sample_size = 1000
#fname = None
fname = f'R{R:.2f}/run{run}_start'
vmc.run(start,stop,fname=fname,save_every=10)
if RANK>0:
    exit()
psi = vmc.psi
x = psi.get_x()
np.save(f'R{R:.2f}/run{run}_start{stop}.npy',x)

