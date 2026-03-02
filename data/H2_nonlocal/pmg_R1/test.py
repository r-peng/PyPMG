import numpy as np
import pickle,h5py,itertools
from PyPMG.pmg_u1 import * 
mo = np.array([[ 0.704936,  0.0553673], #c1
               [ 0.0137269,-0.17477],   #c2
               [-0.704936, -0.0553673], #c3
               [ 0.0770886,-0.981491]]) #c4
mo[:,0] /= np.linalg.norm(mo[:,0])
mo[:,1] -= np.dot(mo[:,0],mo[:,1])*mo[:,0]
mo[:,1] /= np.linalg.norm(mo[:,1])
print('orth=',np.linalg.norm(np.eye(2)-np.dot(mo.T,mo)))
h = np.pi/4
psi = H2State(h,np.ones(6))
psi.right = mo

R = 1
f = h5py.File(f'../sto3g/h2_sto3g_{R:.2f}.h5','r')
const = f['ecore'][()]
eri = f['eri_oao'][:] 
hcore = f['hcore_oao'][:]
f.close()
nao = hcore.shape[0]
#for i,j in itertools.product(range(nao),repeat=2):
#    if abs(hcore[i,j])>1e-6:
#        print(i,j,hcore[i,j])
for i,j,k,l in itertools.product(range(nao),repeat=4):
    if abs(eri[i,j,k,l])>1e-6:
        print(i,j,k,l,eri[i,j,k,l])
#exit()

nso = nao * 2
h = np.zeros((nso,nso))
h[::2,::2] = h[1::2,1::2] = hcore
#for i,j in itertools.product(range(nso),repeat=2):
#    if abs(h[i,j])>1e-6:
#        print(i,j,h[i,j])
eri = eri.transpose(0,2,1,3) # permute to physicist notation (b1,b2,k1,k2)
v = np.zeros((nso,nso,nso,nso))
v[::2,::2,::2,::2] = eri.copy()
v[1::2,1::2,1::2,1::2] = eri.copy()
v[::2,1::2,::2,1::2] = eri.copy()
v[1::2,::2,1::2,::2] = eri.copy()
v_asym = v-v.transpose(0,1,3,2)
v = v_asym/4
#for i,j,k,l in itertools.product(range(nso),repeat=4):
#    if abs(v[i,j,k,l])>1e-6:
#        print(i,j,k,l,v[i,j,k,l])
#exit()

ham = Ham(h,v)
#basis = (1,1,0,0),(1,0,1,0),(0,1,1,0),(1,0,0,1),(0,1,0,1),(0,0,1,1)
basis = (1,1,0,0),(0,1,1,0),(1,0,0,1),(0,0,1,1)
basis_map = {b:i for i,b in enumerate(basis) }
H = np.zeros((len(basis),)*2)
for i,x in enumerate(basis):
    terms = ham.eloc_terms(x)
    for y,coeff in terms.items():
        j = basis_map[y]
        H[j,i] += coeff
print(H)
w,v = np.linalg.eigh(H)
print(w)
print(v)
exit()

vmc = VMC(psi,ham)
E = 0
nsq = 0
for x in itertools.product((0,1),repeat=4):
    psi_x,ex,_ = vmc.eloc(x,derivative=False)
    px = psi_x**2
    nsq += px
    E += px*ex
print('etot=',E,E+const)
