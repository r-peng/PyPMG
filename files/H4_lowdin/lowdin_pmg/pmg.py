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
rho_swap = 0.
run = 22
U0 = True
start,stop = 0,50
optimizer = 'RGN'
rate1 = 0.1
rate2 = 2 
if start==0:
    optimizer = 'SR'
eigen_thresh = 1e-3
#eigen_thresh = None
penalty = False 

R = 1.01
if U0:
    U0 = np.zeros((8,8))
    U0[::2,::2] = U0[1::2,1::2] = np.array([[1, 1, 1, 1.],
                                            [1, 1,-1,-1],
                                            [1,-1, 1,-1],
                                            [1,-1,-1, 1]])/2.
    eps = 0.05
else:
    U0 = None
    eps = 0.5

if start==0:
    x = (np.random.rand(22)*2-1)*eps
    np.save(f'run{run}_start{start}.npy',x)
else:
    x = np.load(f'run{run}_start{start}.npy')
psi = H4MinimalState(x,rho_swap=rho_swap,propose_by=propose_by)

ham = dict()
f = h5py.File(f'../lowdin/h4_{R:.2f}.h5','r')
const = f['ecore'][()]
eri = f['eri_oao'][:] 
hcore = f['hcore_oao'][:]
f.close()
#w,v = np.linalg.eigh(hcore)
#print(w)
#print(v)
#Y = scipy.linalg.logm(v)
#print(Y.real)
#print(Y.imag)
#exit()
#print(hcore)
#for i,j,k,l in itertools.product(range(4),repeat=4):
#    if np.fabs(eri[i,j,k,l])>1e-6:
#        print(i,j,k,l,eri[i,j,k,l])
#exit()

ham['energy'] = QCHamiltonian(hcore,eri)
if penalty:
    ham['S^2'] = TotalSpin(hcore.shape[0],weight=0.1)

sampler = DenseSampler(exact=True)
#sampler = MHSampler(burn_in=20,every=20)
sampler.cf = (1,1,0,0)
#vmc = SGD(psi,ham,sampler)
if optimizer=='SR':
    vmc = SR(psi,ham,sampler)
elif optimizer=='RGN':
    vmc = RGN(psi,ham,sampler)
else:
    raise ValueError
vmc.eigen_thresh = eigen_thresh 
vmc.rate1 = 0.1
vmc.rate2 = rate2
vmc.cond1 = 1e-3
vmc.cond2 = 1e-3
vmc.sample_size = 1000
vmc.run(start,stop,save_wfn=False)
if RANK>0:
    exit()
psi = vmc.psi
for pmg in psi.pmg_ls:
    print(pmg.x)
#U = psi.pmg_ls[-1].Y['expY']
x = psi.get_x()
np.save(f'run{run}_start{stop}.npy',x)
