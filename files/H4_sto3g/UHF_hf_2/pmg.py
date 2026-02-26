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
paths = [] 
for l3 in (0,2):
    choices = {0:(0,1,3),2:(1,2,5)}[l3]
    for l2 in choices:
        choices_ = {0:(0,1),1:(0,2),2:(0,3),3:(1,2),5:(2,3)}[l2]
        for l1 in choices_:
            path = 0,l3,l2,l1
            paths.append(path)
            print(path)
path,nparam = 0,24
path,nparam = 1,24
path,nparam = 2,24
path,nparam = 3,24
path,nparam = 4,24
path,nparam = 5,24
path,nparam = 6,24
path,nparam = 7,24
path,nparam = 8,24
path,nparam = 9,24

run = 2 
start,stop = 100,150
penalty = False 
optimizer = 'RGN'
rate2 = 1 
eigen_thresh = 1e-3
if start==0:
    optimizer = 'SR'
x = np.load(f'path{path}_run{run}_start{start}.npy')
#for i in range(5):
#    x = np.random.rand(nparam)*2-1
#    np.save(f'path{path}_run{i}_start0.npy',x)
psi = H4MinimalHF(x,path=paths[path],rho_swap=rho_swap,propose_by=propose_by)
#exit()

ham = dict()
R = 1.01
f = h5py.File(f'../hf_mo/h4_{R:.2f}.h5','r')
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
np.save(f'path{path}_run{run}_start{stop}.npy',x)
