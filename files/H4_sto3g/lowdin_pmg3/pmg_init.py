import numpy as np
import scipy.linalg
import pickle,h5py,itertools
from PyPMG.h4_model import * 
from PyPMG.vmc import *
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(precision=10,suppress=True)

propose_by = 'uniform'
rho_swap = 0
run = 0
U0 = True
start,stop = 0,1
optimizer = 'SR'
#optimizer = 'LM'
rate1 = 0.1
rate2 = 5
if optimizer=='LM':
    rate2 = 0.5
#if start==0:
#    optimizer = 'SR'
#eigen_thresh = 1e-6
eigen_thresh = None
penalty = False 
HF_typ = 'GHF'
symmetry = 'u1'
pmg_typ = 3 
remove_redundant = True

R = 1.01
dR = 0.02
if U0:
    U0 = np.zeros((8,8))
    U0[::2,::2] = U0[1::2,1::2] = np.array([[1, 1, 1, 1.],
                                            [1, 1,-1,-1],
                                            [1,-1, 1,-1],
                                            [1,-1,-1, 1]])/2.
    eps = 0.1
    #eps = 1
else:
    U0 = None
    eps = 0.5
psi = H4MinimalState(HF_typ,pmg_typ=pmg_typ,U0=U0,remove_redundant=remove_redundant,
                     symmetry=symmetry,rho_swap=rho_swap,propose_by=propose_by)

if start==0:
    #x = (np.random.rand(psi.nparam)*2-1)*eps
    #COMM.Bcast(x,root=0)
    x = np.load(f'R{R:.2f}/run{run}_start{start}.npy') 
    #if RANK==0:
    #    np.save(f'R{R:.2f}/run{run}_start{start}.npy',x)
else:
    x = np.load(f'R{R:.2f}/run{run}_start{start}.npy')
psi._update(x)

ham = dict()
f = h5py.File(f'../lowdin/h4_{R:.2f}.h5','r')
const = f['ecore'][()]
eri = f['eri_oao'][:] 
hcore = f['hcore_oao'][:]
f.close()
eri = np.zeros_like(eri)

ham['energy'] = QCHamiltonian(hcore,eri)
if penalty:
    ham['S^2'] = TotalSpin(hcore.shape[0],weight=0.1)

#if RANK==0:
#    hcore = np.zeros((psi.nsite,)*2)
#    hcore[::2,::2] = ham['energy'].hcore
#    hcore[1::2,1::2] = ham['energy'].hcore
#    E = 0
#    for x in itertools.product((0,1),repeat=len(psi.decimated)):
#        E += psi.compute_energy(x,hcore,eri=ham['energy'].eri)
#    print('energy by summing decimated indices=',E)
#exit()

sampler = DenseSampler(exact=True)
#sampler = MHSampler(burn_in=20,every=20)
sampler.cf = (1,1,0,0)
if optimizer=='SGD':
    vmc = SGD(psi,ham,sampler)
elif optimizer=='SR':
    vmc = SR(psi,ham,sampler)
elif optimizer=='RGN':
    vmc = RGN(psi,ham,sampler)
elif optimizer=='LM':
    vmc = LM(psi,ham,sampler)
else:
    raise ValueError
vmc.eigen_thresh = eigen_thresh 
vmc.rate1 = rate1 
vmc.rate2 = rate2
vmc.cond1 = 1e-3
vmc.cond2 = 1e-3
vmc.sample_size = 1000
vmc.run(start,stop,fname=None)
if RANK>0:
    exit()
psi = vmc.psi
for pmg in psi.pmg_ls:
    print(pmg.x)
#U = psi.pmg_ls[-1].Y['expY']
#x = psi.get_x()
#np.save(f'R{R:.2f}/run{run}_start{stop}.npy',x)
