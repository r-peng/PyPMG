import numpy as np
import scipy.linalg
import pickle,h5py,itertools
from PyPMG.hchain import * 
from PyPMG.jastrow import * 
from PyPMG.vmc import *
np.set_printoptions(precision=5,suppress=True,threshold=100000)

propose_by = 'uniform'
rho_swap = 0.
thresh = 1e-28
run = 0 
every = 50
if RANK==0:
    print('every=',every)

U0 = True
start = 0
symmetry = 'u11'
HF_typ = 'GHF'
pmg_typ = 2 

nsites = 50,50
nelec = 25,25
nsite = sum(nsites)
R = 2.0
jas = Jastrow(nsite=nsite,Jmax=None)
if U0:
    U0 = np.zeros((100,100))
    mo = np.load(f'../../lowdin_hfmo/r{R:.2f}.npy')
    U0[::2,::2] = mo[0]
    U0[1::2,1::2] = mo[1]
    eps = 0.1
else:
    U0 = None
    eps = 0.5
pmg = get_h50_minimum(HF_typ,pmg_typ,manual_derivative=False,remove_redundant=True,U0=U0,symmetry=symmetry,rho_swap=rho_swap,propose_by=propose_by,thresh=thresh)

psi = JastrowPMGState(jas,pmg)
x = np.load(f'run{run}_start{start}.npy')
psi.update(x)

sampler = MHSampler(burn_in=0,every=every)
occ = get_occ_from_mo(mo[0],nelec[0])
occ = [2*int(p) for p in occ]+[2*int(p)+1 for p in occ]
sampler.cf = get_config_from_occ(occ,nsite)
sampler.px = psi.log_prob(sampler.cf)
print(sampler.cf,sampler.px)

burn_in = 20
print('burn in...')
for i in range(burn_in):
    cf,omega = sampler.sample(psi,iprint=1)
    print(f'{i}/{burn_in},cf={cf},px={omega}')
ncf = 100
cfs = [None] * ncf
print('generating configs...')
for i in range(ncf):
    cfs[i],omega = sampler.sample(psi,iprint=1)
    print(f'{i}/{ncf},cf={cf},px={omega}')
np.save(f'run{run}_start{start}_configs.npy',np.array(cfs))

