import numpy as np
import pickle,h5py,itertools
from PyPMG.pmg import * 
from PyPMG.vmc import *
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(precision=10,suppress=True)

h = np.pi/4
k12 = -1.3945
kvec = np.array([np.pi/4,k12])
psi = H2State(x[0],x[1:])
#basis = (1,1,0,0),(1,0,0,1),(0,1,1,0),(0,0,1,1)
#for i in range(1,401):
#    x = np.load(f'psi{i}.npy')
#    psi = H2State(x[0],x[1:])
#    psi_x = [psi.amplitude(x) for x in basis]
#    print(i,psi_x)
#exit()

R = 1
f = h5py.File(f'../sto3g/h2_sto3g_{R:.2f}.h5','r')
const = f['ecore'][()]
eri = f['eri_oao'][:] 
hcore = f['hcore_oao'][:]
f.close()
ham = QCHamiltonian(hcore,eri)

sampler = DenseSampler(exact=True)
#sampler = MHSampler(burn_in=20,every=10)
sampler.cf = (1,1,0,0)
#vmc = SGD(psi,ham,sampler)
vmc = SR(psi,ham,sampler)
vmc.rate1 = 0.1
vmc.cond1 = 1e-3
vmc.sample_size = 1000
vmc.run(0,1,save_wfn=False)
if RANK>0:
    exit()
psi = vmc.sampler.psi
print(psi.h,np.pi/4)
print(psi.kvec)
print(psi.right)
