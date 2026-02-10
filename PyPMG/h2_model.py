import numpy as np
import scipy,itertools
import torch
from PyPMG.pmg import * 
class H2MinimalState(PMGState):
    def __init__(self,x,**kwargs):
        # modes order: c1,c2,c3,c4
        nsites = 2,2
        nelec = 1,1

        pmg_ls = []

        nsite = sum(nsites)
        hop_ls = (0,2),
        def fxn(cf,x):
            return 1-2*cf[1]
        pmg_ls.append(PMG(nsite,hop_ls,fxn=fxn))

        hop_ls = (0,2),(1,3)
        pmg_ls.append(MG(nsites,hop_ls=hop_ls))
        super().__init__(nsites,nelec,pmg_ls,**kwargs)

        self._update(x)
def comm(A,B):
    return np.dot(A,B)-np.dot(B,A)
class H2MinimalHamiltonian(QCHamiltonian):
    def _2spin(self,H):
        a = (H[0,0]-H[3,3])/4
        b = H[0,3]
        c = (H[0,0]-2*a-H[1,1])/2
        d = (H[0,0]-2*a+H[1,1])/2
        print('a,b,c,d=',a,b,c,d)


        sx = np.array([[0,1],[1,0]]) # 2Jx
        sy = np.array([[0,-1j],[1j,0]]) # 2Jy
        sz = np.array([[1,0],[0,-1]]) # 2Jz
        I = np.eye(2)
        IMB = np.einsum('ij,kl->ikjl',I,I).reshape(4,4) 
        N = 2

        z1 = np.einsum('ij,kl->ikjl',sz,I).reshape(4,4)
        z2 = np.einsum('ij,kl->ikjl',I,sz).reshape(4,4)
        z12 = z1+z2
        xx = np.einsum('ij,kl->ikjl',sx,sx).reshape(4,4)
        zz = np.einsum('ij,kl->ikjl',sz,sz).reshape(4,4)
        #print(a*z12+b*xx+c*zz+d*IMB)
        #print(comm(sx/2,sy/2))
        #print(comm(sy/2,sz/2))
        #print(comm(sz/2,sx/2))

        h1 = np.arctan(b/2/a)/2
        h2 = np.pi/4 
        yx = np.einsum('ij,kl->ikjl',1j*sy,sx).reshape(4,4)
        for h in (h1,h2):
            U = scipy.linalg.expm(-h*yx)
            print('h=',h) 
            print('H conj=')
            print(np.dot(U.T.conj(),np.dot(H,U)))

