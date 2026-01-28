import numpy as np
import scipy,itertools
import torch
from PyPMG.fermion_state import * 
def get_MG_MB(basis,ix1,ix2,theta):
    basis_map = {b:i for i,b in enumerate(basis)}
    kappa = np.zeros((len(basis),)*2)
    def fill(i,ops,coeff):
        x = basis[i]
        y,sign = string_act(x,ops)
        if y is not None:
            j = basis_map[tuple(y)]
            kappa[j,i] += sign*coeff
    for i in range(len(basis)):
        ops = (ix1,'cre'),(ix2,'des')
        fill(i,ops,theta)
        ops = (ix2,'cre'),(ix1,'des')
        fill(i,ops,-theta)
    return kappa
class H2State(FermionState):
    def __init__(self,x,**kwargs):
        # modes order: c1,c2,c3,c4
        super().__init__((2,2),(1,1),**kwargs)

        self.nparam = 3
        assert len(x)==3

        self.hmat = np.zeros((4,4))
        self.hmat[0,2] = 1
        self.hmat -= self.hmat.T 

        self.k02 = np.zeros((4,4))
        self.k02[0,2] = 1 
        self.k02 -= self.k02.T

        self.k13 = np.zeros((4,4))
        self.k13[1,3] = 1 
        self.k13 -= self.k13.T

        self._update(x)
    def get_x(self,h=None,kvec=None):
        if h is None:
            h = self.h
        if kvec is None:
            kvec = self.kvec
        return np.concatenate([np.ones(1)*h,kvec])
    def _update(self,x):
        self.h = x[0] 
        self.left = scipy.linalg.expm(-self.h*self.hmat)
        print(self.left)
        exit()

        self.kvec = x[1:] 
        k = self.k02*self.kvec[0]+self.k13*self.kvec[1]
        self.right = scipy.linalg.expm(k)[:,:2]
    def _amplitude_and_derivative(self,x):
        h = torch.tensor(self.h,requires_grad=True)
        kvec = torch.tensor(self.kvec,requires_grad=True)
        hmat = torch.tensor(self.hmat,requires_grad=False)
        k02 = torch.tensor(self.k02,requires_grad=False)
        k13 = torch.tensor(self.k13,requires_grad=False)

        h_ = h*(-1)**x[1]
        left = torch.linalg.matrix_exp(-h_*hmat)
        idx = np.argwhere(x).flatten()
        left = left[:,idx]

        k = k02*kvec[0]+k13*kvec[1]
        right = torch.linalg.matrix_exp(k)[:,:2]

        psi_x = torch.matmul(left.T,right)
        psi_x = torch.linalg.det(psi_x)

        psi_x.backward()
        vx = self.get_x(h=h.grad.numpy(force=True),kvec=kvec.grad.numpy(force=True))
        psi_x = psi_x.numpy(force=True)
        return psi_x,vx
    def _amplitude(self,x):
        left = self.left if x[1]==0 else self.left.T
        idx = np.argwhere(x).flatten()
        left = left[:,idx]
        M = np.dot(left.T,self.right)
        return np.linalg.det(M)
    def get_PMG_MB(self,basis):
        basis_map = {b:i for i,b in enumerate(basis)}
        kappa = np.zeros((len(basis),)*2)
        def fill(i,ops,coeff):
            x = basis[i]
            y,sign = string_act(x,ops)
            if y is not None:
                j = basis_map[tuple(y)]
                kappa[j,i] += sign*coeff

        for i in range(len(basis)):
            ops = (0,'cre'),(2,'des')
            fill(i,ops,self.h)
            ops = (0,'cre'),(1,'cre'),(1,'des'),(2,'des')
            fill(i,ops,-2*self.h)

            ops = (2,'cre'),(0,'des')
            fill(i,ops,-self.h)
            ops = (2,'cre'),(1,'cre'),(1,'des'),(0,'des')
            fill(i,ops,2*self.h)
        return kappa
    def get_13_MB(self,basis):
        return -get_MG_MB(basis,0,2,self.kvec[0])
    def get_24_MB(self,basis):
        return -get_MG_MB(basis,1,3,self.kvec[1])
class H2State_GHF(H2State):
    def __init__(self,x,**kwargs):
        super().__init__(x,**kwargs)
        self.nparam -= 1
    def get_x(self,h=None,kvec=None):
        return super().get_x(h=h,kvec=kvec)[1:]
    def update(self,x):
        super()._update(self.h,x)
def comm(A,B):
    return np.dot(A,B)-np.dot(B,A)
class H2Hamiltonian(QCHamiltonian):
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

if __name__=='__main__':
    N = 4
    h = np.random.rand()
    kvec = np.random.rand(N*(N-1)//2)
    psi = H2State(h,kvec)
    for i,j,k,l in itertools.product((0,1),repeat=4):
        x = i,j,k,l
        amp1 = psi.amplitude(x)
        amp2,_ = psi.amplitude_and_derivative(x)
        err = np.fabs(amp1-amp2)
        if err>1e-10:
            print(x,amp1,amp2)

