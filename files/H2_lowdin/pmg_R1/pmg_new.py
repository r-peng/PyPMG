import numpy as np
import pickle,h5py,itertools
from PyPMG.h2_model import * 
from PyPMG.vmc import *
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(precision=10,suppress=True)

propose_by = 'uniform'
rho_swap = 0.
x = .1,.2,.13
psi = H2MinimalState(x,rho_swap=rho_swap,propose_by=propose_by)

ham = dict()
R = 1.
f = h5py.File(f'../lowdin/h2_{R:.2f}.h5','r')
const = f['ecore'][()]
eri = f['eri_oao'][:] 
hcore = f['hcore_oao'][:]
f.close()
#for i,j,k,l in itertools.product((0,1),repeat=4):
#    if np.fabs(eri[i,j,k,l])>1e-6:
#        print(i,j,k,l,eri[i,j,k,l])

ham['energy'] = QCHamiltonian(hcore,eri)
#ham['S^2'] = TotalSpin(hcore.shape[0],weight=0.1)

start,stop = 0,400
sampler = DenseSampler(exact=True)
#sampler = MHSampler(burn_in=20,every=20)
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
print(psi.get_x())
for pmg in psi.pmg_ls:
    print(pmg.Y['expY'])
#x = psi.get_x()
#np.save('x_100.npy',x)
