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
path = 0 
start,stop = 0,50
optimizer = 'RGN'
rate1 = 0.1
rate2 = 2 
eigen_thresh = 1e-3
#eigen_thresh = None
penalty = False 
HF_typ = 'GHF'
symmetry = 'u11'
pmg_typ = 2 

R = 1.23
dR = 0.02
U0 = np.zeros((8,8))
U0[::2,::2] = U0[1::2,1::2] = np.array([[1, 1, 1, 1.],
                                        [1, 1,-1,-1],
                                        [1,-1, 1,-1],
                                        [1,-1,-1, 1]])/2.
psi = H4MinimalState(HF_typ,pmg_typ=pmg_typ,U0=U0,symmetry=symmetry,rho_swap=rho_swap,propose_by=propose_by)

#x = np.load(f'R1.01/run4_start50.npy') # path0 
#x = np.load(f'R1.25/run0_start50.npy') # path1
x = np.load(f'path{path}/R{R:.2f}.npy') 
psi._update(x)

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
vmc.run(start,stop,fname=None)
if RANK>0:
    exit()
psi = vmc.psi
for pmg in psi.pmg_ls:
    print(pmg.x)
#U = psi.pmg_ls[-1].Y['expY']
x = psi.get_x()
np.save(f'path{path}/R{R:.2f}.npy',x)
