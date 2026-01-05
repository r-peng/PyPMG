import numpy as np
import pickle,h5py,itertools
from PyPMG.pmg_u1 import H2State,Ham 
from PyPMG.vmc import *

h = np.random.rand()
kvec = np.random.rand(6)*2-1
psi = H2State(h,kvec)

R = 1
f = h5py.File(f'../sto3g/h2_sto3g_{R:.2f}.h5','r')
const = f['ecore'][()]
eri = f['eri_oao'][:] 
hcore = f['hcore_oao'][:]
f.close()

nao = hcore.shape[0]
nso = nao * 2
h = np.zeros((nso,nso))
h[::2,::2] = h[1::2,1::2] = hcore
eri = eri.transpose(0,2,1,3) # permute to physicist notation (b1,b2,k1,k2)
v = np.zeros((nso,nso,nso,nso))
v[::2,::2,::2,::2] = eri.copy()
v[1::2,1::2,1::2,1::2] = eri.copy()
v[::2,1::2,::2,1::2] = eri.copy()
v[1::2,::2,1::2,::2] = eri.copy()
v_asym = v-v.transpose(0,1,3,2)
v = v_asym/4
ham = Ham(h,v)

#sampler = DenseSampler(exact=True)
sampler = MHSampler(burn_in=10)
sampler.cf = np.array([1,0,0,1]) 
#vmc = SGD(psi,ham,sampler)
vmc = SR(psi,ham,sampler)
vmc.rate1 = 0.1
vmc.cond1 = 1e-3
vmc.sample_size = 1000
vmc.run(0,200)
