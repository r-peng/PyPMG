import numpy as np
import scipy
import pickle,h5py,itertools
from PyPMG.pmg import * 
from PyPMG.vmc import *
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=10,linewidth=10000)
#U = np.array(
#[[ 0.69499906  ,  0.1302933938, -0.3932004122,  0.587701773 ],
# [ 0.0322848543, -0.1723098066, -0.8182746962, -0.5474427278],
# [-0.6950040678, -0.1302662046, -0.3931827951,  0.5877136645],
# [ 0.1813927798, -0.9676588645,  0.1456957344,  0.0974973558]]
#)
#print('tan1=',U[2,0]/U[0,0],U[2,1]/U[0,1],-U[0,2]/U[2,2],-U[0,3]/U[2,3])
#print('tan2=',U[3,0]/U[1,0],U[3,1]/U[1,1],-U[1,2]/U[3,2],-U[1,3]/U[3,3])
#print('tan3=',-U[0,1]/U[0,0],U[1,0]/U[1,1],-U[0,3]/U[0,2],U[1,2]/U[1,3])
#exit()

R = 1
f = h5py.File(f'../sto3g/h2_sto3g_{R:.2f}.h5','r')
const = f['ecore'][()]
eri = f['eri_oao'][:] 
hcore = f['hcore_oao'][:]
f.close()

ham = QCHamiltonian(hcore,eri)
basis = (1,1,0,0),(1,0,1,0),(0,1,1,0),(1,0,0,1),(0,1,0,1),(0,0,1,1)
Hmat,_ = ham.get_MB_hamiltonian(basis=basis)
if RANK==0:
    print('Hmat')
    print(Hmat)
    w,v = np.linalg.eigh(Hmat)
    print('H eigvals=',w)

check_mo = False
x = np.load('../pmg/psi_init.npy')[:4]
h,kvec = x[0],x[1:]
h = np.pi/4
if RANK==0:
    print('h=',h,np.pi/4)
psi = H2State_GHF(h,kvec)
if check_mo:
    mo = np.array([[ 0.704936,  0.0553673], #c1
                   [ 0.0137269,-0.17477],   #c2
                   [-0.704936, -0.0553673], #c3
                   [ 0.0770886,-0.981491]]) #c4
    mo[:,0] /= np.linalg.norm(mo[:,0])
    mo[:,1] -= np.dot(mo[:,0],mo[:,1])*mo[:,0]
    mo[:,1] /= np.linalg.norm(mo[:,1])
    if RANK==0:
        print('orth=',np.linalg.norm(np.eye(2)-np.dot(mo.T,mo)))
    psi.right = mo
kappa = psi.get_PMG_MB(basis)
U = scipy.linalg.expm(kappa)
if RANK==0:
    print('U')
    print(U)
Hmat = np.dot(U.T,np.dot(Hmat,U))
Ham = MBHamiltonian(Hmat,basis)
if RANK==0:
    w,v = np.linalg.eigh(Hmat)
    print('U^THU E0=',w[0])
#exit()

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
