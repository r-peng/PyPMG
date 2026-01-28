import numpy as np
import scipy
import pickle,h5py,itertools
from PyPMG.h2_model import * 
from PyPMG.vmc import *
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=10,linewidth=10000)
R = 1.
f = h5py.File(f'../lowdin/h2_{R:.2f}.h5','r')
const = f['ecore'][()]
eri = f['eri_oao'][:] 
hcore = f['hcore_oao'][:]
f.close()

basis = (1,1,0,0),(1,0,1,0),(0,1,1,0),(1,0,0,1),(0,1,0,1),(0,0,1,1)
basis = (1,1,0,0),(1,0,0,1),(0,1,1,0),(0,0,1,1),(1,0,1,0),(0,1,0,1)
#basis = (1,1,0,0),(0,0,1,1),(1,0,0,1),(0,1,1,0),(1,0,1,0),(0,1,0,1)
nsite = 4
#Sz = get_MB_Sz(nsite,basis)
#Sp = get_MB_Spm(nsite,basis,'+')
#Sm = get_MB_Spm(nsite,basis,'-')
#Ssq = np.dot(Sz,Sz)+(np.dot(Sp,Sm)+np.dot(Sm,Sp))/2
#S2,_ = TotalSpin(2).get_MB_matrix(basis=basis)
#print(Ssq)
#print(S2)
#exit()

ham = H2Hamiltonian(hcore,eri)
Hmat,_ = ham.get_MB_matrix(basis=basis)
if RANK==0:
    print('Hmat')
    print(Hmat)
    w,v = np.linalg.eigh(Hmat)
    print('H eigvals=',w)
#    comm = np.dot(Hmat,Ssq)-np.dot(Ssq,Hmat)
#    print('[H,S^2]=',np.linalg.norm(comm))

x = np.array([0,np.pi/4,np.pi/4])
#if RANK==0:
#    print('h=',h,np.pi/4)
psi = H2State(x)
print('kvec=',psi.kvec)
kappa = psi.get_13_MB(basis)
U13 = scipy.linalg.expm(kappa)
print('U13')
print(U13)
kappa = psi.get_24_MB(basis)
U24 = scipy.linalg.expm(kappa)
print('U24')
print(U24)
U = np.dot(U13,U24)
print('Utotal')
print(U)
Hmat = np.dot(U.T,np.dot(Hmat,U))
print('U^THU')
print(Hmat)
Hmat = Hmat[:4,:4]
Hmat[1,2] = -Hmat[1,2]
Hmat[2,1] = -Hmat[2,1]
print(Hmat)
w,v = np.linalg.eigh(Hmat)
#print(v)
print(w)
#H1,H2 = Hmat[:2,:2],Hmat[2:,2:]
#for H in (H1,H2):
#    w,v = scipy.linalg.eigh(H)
#    print(w)
#    print(v)
#    print(np.arctan(v[1,0]/v[0,0]))

ham._2spin(Hmat)
exit()
Ham = MBHamiltonian(Hmat,basis)

if check_mo:
    vmc = VMC(psi,ham)
    E = 0
    nsq = 0
    for x in itertools.product((0,1),repeat=4):
        psi_x,ex,_ = vmc.eloc(x,derivative=False)
        px = psi_x**2
        nsq += px
        E += px*ex
    print('etot=',E,E+const)
else:
    sampler = DenseSampler(exact=True)
    vmc = SR(psi,ham,sampler)
    vmc.rate1 = 0.1
    vmc.cond1 = 1e-4
    vmc.run(0,500)
    if RANK==0:
        psi = vmc.sampler.psi
        print(psi.h,np.pi/4)
        print(psi.right)
