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
#kvec = np.random.rand(3)*2-1
#psi = H2State(h,kvec)
#x = psi.get_x()
#np.save('init.npy',x)
#exit()

#x = np.load('init.npy')[:3]
#x[:2] = np.pi/4
#x = np.array([0,np.pi/4,np.pi/4])

#x = np.array([0,0.1,0.1])
propose_by = 'uniform'
start,stop = 0,100

x = np.load('x_100.npy')
propose_by = 'hamiltonian'
start,stop = 100,400

rho_swap = 0.
psi = H2State(x,rho_swap=rho_swap,propose_by=propose_by)
#basis = (1,1,0,0),(1,0,0,1),(0,1,1,0),(0,0,1,1)
#for i in range(1,401):
#    x = np.load(f'psi{i}.npy')
#    psi = H2State(x[0],x[1:])
#    psi_x = [psi.amplitude(x) for x in basis]
#    print(i,psi_x)
#exit()

ham = dict()
R = 1
f = h5py.File(f'../sto3g/h2_sto3g_{R:.2f}.h5','r')
const = f['ecore'][()]
eri = f['eri_oao'][:] 
hcore = f['hcore_oao'][:]
f.close()
ham['energy'] = QCHamiltonian(hcore,eri)
#ham['S^2'] = TotalSpin(hcore.shape[0],weight=0.1)

#sampler = DenseSampler(exact=True)
sampler = MHSampler(burn_in=20,every=20)
sampler.cf = (1,1,0,0)
#vmc = SGD(psi,ham,sampler)
vmc = SR(psi,ham,sampler)
vmc.rate1 = 0.1
vmc.cond1 = 1e-3
vmc.sample_size = 1000
vmc.run(start,stop,save_wfn=False)
if RANK>0:
    exit()
psi = vmc.psi
print(psi.h,np.pi/4)
print(psi.kvec)
print(psi.right)
#x = psi.get_x()
#np.save('x_100.npy',x)
