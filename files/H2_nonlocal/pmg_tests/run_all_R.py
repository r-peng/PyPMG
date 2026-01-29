import numpy as np
import scipy
import pickle,h5py,itertools
from PyPMG.pmg import * 
np.set_printoptions(suppress=True,precision=10,linewidth=10000)

#basis = (1,1,0,0),(1,0,1,0),(0,1,1,0),(1,0,0,1),(0,1,0,1),(0,0,1,1)
basis = (1,1,0,0),(1,0,0,1),(0,1,1,0),(0,0,1,1),(1,0,1,0),(0,1,0,1)
h = np.pi/4
kvec = np.array([np.pi/4,1])
psi = H2State_GHF(h,kvec)
for R in np.arange(1,2.01,.01):
    print(f'##################### R={R:.2f} #############################')
    f = h5py.File(f'../sto3g/h2_sto3g_{R:.2f}.h5','r')
    const = f['ecore'][()]
    eri = f['eri_oao'][:] 
    hcore = f['hcore_oao'][:]
    f.close()
    print('ecore=',const)

    ham = QCHamiltonian(hcore,eri)
    Hmat,_ = ham.get_MB_hamiltonian(basis=basis)
    print('Hmat')
    print(Hmat)
    w,v = np.linalg.eigh(Hmat)
    print('w=',w)

    kappa = psi.get_PMG_MB(basis)
    U = scipy.linalg.expm(kappa)
    Hmat = np.dot(U.T,np.dot(Hmat,U))

    kappa = psi.get_13_MB(basis)
    U = scipy.linalg.expm(kappa)
    Hmat = np.dot(U.T,np.dot(Hmat,U))
    print('U1^TU^THUU1')
    print(Hmat)
    Hmat = Hmat[:4,:4]
    H1,H2 = Hmat[:2,:2],Hmat[2:,2:]
    for i,H in enumerate((H1,H2)):
        w,v = scipy.linalg.eigh(H)
        print(f'E{i}=',w[0])
        print(v)
        print(f'theta{i}=',np.arctan(v[1,0]/v[0,0]))
