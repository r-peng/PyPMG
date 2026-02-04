import numpy as np
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
pairs = 'GHF'
if pairs=='GHF':
    pairs = (0,1),(0,2),(0,4),(1,3),(1,5),(1,6),(2,4),(2,5),(2,6),(3,5),(3,7),(4,6),(4,7),(5,7),(6,7)
#x = np.random.rand(28+3)
#np.save('x_init.npy',x)
#exit()
start,stop = 0,100
if start==0:
    x = np.load('x_init.npy')[:18]
else:
    x = np.load(f'{pairs}/x_{start}_{pairs}.npy')
psi = H4MinimalState(x,pairs=pairs,rho_swap=rho_swap,propose_by=propose_by)

ham = dict()
R = 1.01
f = h5py.File(f'lowdin/h4_{R:.2f}.h5','r')
const = f['ecore'][()]
eri = f['eri_oao'][:] 
hcore = f['hcore_oao'][:]
f.close()
print(hcore)
for i,j,k,l in itertools.product(range(4),repeat=4):
    if np.fabs(eri[i,j,k,l])>1e-6:
        print(i,j,k,l,eri[i,j,k,l])
exit()

ham['energy'] = QCHamiltonian(hcore,eri)
#ham['S^2'] = TotalSpin(hcore.shape[0],weight=0.1)

sampler = DenseSampler(exact=True)
#sampler = MHSampler(burn_in=20,every=20)
sampler.cf = (1,1,0,0)
#vmc = SGD(psi,ham,sampler)
vmc = SR(psi,ham,sampler)
vmc.eigen_thresh = 1e-3
vmc.rate1 = 0.1
vmc.cond1 = 1e-3
vmc.sample_size = 1000
vmc.run(start,stop,save_wfn=False)
if RANK>0:
    exit()
psi = vmc.psi
x = psi.get_x()
print(x)
print(psi.mg.mo)
np.save(f'{pairs}/x_{stop}_{pairs}.npy',x)
