import numpy as np
import scipy.linalg
import pickle,h5py,itertools
from PyPMG.pmg import * 
from PyPMG.vmc import *
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(precision=10,suppress=True)

propose_by = 'uniform'
rho_swap = 0.
run = 2 
U0 = True
start,stop = 150,200
optimizer = 'RGN'
#optimizer = 'SR'
rate1 = 0.05
rate2 = 1
eigen_thresh = 1e-3
#eigen_thresh = None
penalty = False 
if start==0:
    optimizer = 'SR'
    rate1 = 0.05

R = 1.21
#dR = 0.02
#Rprev = R-dR
#try:
#    x = np.load(f'x_R{R:.2f}.npy')
#except FileNotFoundError:
#    x = np.load(f'x_R{Rprev:.2f}.npy')

nsites = 4,4
nelec = 2,2
nsite = sum(nsites)
jas_ls = [Jastrow(nsite=nsite,Jmax=None)]
pmg_ls = [MG(nsites,hop_ls='GHF')]
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
psi = JastrowPMGState(nsites,nelec,pmg_ls,jas_ls,U0=U0,rho_swap=rho_swap,propose_by=propose_by)
if start==0:
    x = (np.random.rand(psi.nparam)*2-1)*eps
    np.save(f'R{R:.2f}/run{run}_start{start}.npy',x)
    #x = np.load(f'R{R:.2f}/run{run}_start{start}.npy')
else:
    x = np.load(f'R{R:.2f}/run{run}_start{start}.npy')
psi._update(x)

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
fname = None
#fname = f'x_R{R:.2f}.npy'
vmc.run(start,stop,fname=fname)
if RANK>0:
    exit()
psi = vmc.psi
for pmg in psi.pmg_ls:
    print(pmg.x)
x = psi.get_x()
np.save(f'R{R:.2f}/run{run}_start{stop}.npy',x)
