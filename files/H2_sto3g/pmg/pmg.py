import numpy as np
import pickle,h5py,itertools
from PyPMG.pmg import * 
from PyPMG.vmc import *
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(precision=10,suppress=True)

#h = np.random.rand()
#kvec = np.random.rand(6)*2-1
#psi = H2State(h,kvec)
#x = psi.get_x()
#np.save('psi_init.npy',x)

x = np.load('psi_init.npy')
psi = H2State(x[0],x[1:])

R = 1
f = h5py.File(f'../sto3g/h2_sto3g_{R:.2f}.h5','r')
const = f['ecore'][()]
eri = f['eri_oao'][:] 
hcore = f['hcore_oao'][:]
f.close()
ham = QCHamiltonian(hcore,eri)

#sampler = DenseSampler(exact=True)
sampler = MHSampler(burn_in=20)
sampler.cf = (1,0,0,1)
#vmc = SGD(psi,ham,sampler)
vmc = SR(psi,ham,sampler)
vmc.rate1 = 0.1
vmc.cond1 = 1e-3
vmc.sample_size = 1000
vmc.run(0,400)
if RANK>0:
    exit()
psi = vmc.sampler.psi
print(psi.h,np.pi/4)
print(psi.right)
