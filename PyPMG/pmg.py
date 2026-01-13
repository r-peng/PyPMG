import numpy as np
import scipy,itertools
import torch
from PyPMG.hamiltonian import * 
class H2State(FermionState):
    def __init__(self,h,kvec,symmetry='u11'):
        # modes order: c1,c2,c3,c4
        super().__init__(4,(1,1),symmetry=symmetry)

        self.nparam = 1+len(kvec)
        self.idxs = [(i,j) for i in range(self.nsite) for j in range(i+1,self.nsite)]
        assert len(self.idxs)==len(kvec)
        self._update(h,kvec)
    def get_x(self,h=None,kvec=None):
        if h is None:
            h = self.h
        if kvec is None:
            kvec = self.kvec
        return np.concatenate([np.ones(1)*h,kvec])
    def update(self,x):
        self._update(x[0],x[1:])
    def _update(self,h,kvec):
        self.hmat = np.zeros((4,4))
        self.hmat[0,2] = 1
        self.hmat[2,0] = -1
        self.h = h
        self.left = scipy.linalg.expm(-h*self.hmat)

        self.kvec = kvec
        self.kappa = np.zeros((4,4))
        for ix,(i,j) in enumerate(self.idxs):
            self.kappa[i,j] = kvec[ix]
        self.right = scipy.linalg.expm(self.kappa-self.kappa.T)
    def amplitude_and_derivative(self,x):
        if self.symmetry=='u1':
            if sum(x)!=sum(self.nelec):
                return 0,None
        if self.symmetry=='u11':
            if sum(x[::2])!=self.nelec[0]:
                return 0,None
            if sum(x[1::2])!=self.nelec[1]:
                return 0,None

        h = torch.tensor(self.h,requires_grad=True)
        #kvec = torch.tensor(self.kvec,requires_grad=True)
        kappa = torch.tensor(self.kappa,requires_grad=True)
        hmat = torch.tensor(self.hmat,requires_grad=False)

        h_ = h*(-1)**x[1]
        left = torch.linalg.matrix_exp(-h_*hmat)
        idx = np.argwhere(x).flatten()
        left = left[:,idx]

        right = torch.linalg.matrix_exp(kappa-kappa.T)

        psi_x = torch.matmul(left.T,right[:,:2])
        psi_x = torch.linalg.det(psi_x)

        psi_x.backward()
        vx = kappa.grad.numpy(force=True)
        vx = np.array([vx[i,j] for i,j in self.idxs])
        vx = self.get_x(h=h.grad.numpy(force=True),kvec=vx)
        psi_x = psi_x.numpy(force=True)
        return psi_x,vx
    def amplitude(self,x):
        if self.symmetry=='u1':
            if sum(x)!=sum(self.nelec):
                return 0,None
        if self.symmetry=='u11':
            if sum(x[::2])!=self.nelec[0]:
                return 0,None
            if sum(x[1::2])!=self.nelec[1]:
                return 0,None

        left = self.left if x[1]==0 else self.left.T
        idx = np.argwhere(x).flatten()
        left = left[:,idx]

        M = np.dot(left.T,self.right[:,:2])
        return np.linalg.det(M)
    def get_PMG_MB(self,basis):
        basis_map = {b:i for i,b in enumerate(basis)}
        kappa = np.zeros((len(basis),)*2)
        def fill(ops,coeff):
            y,sign = string_act(x,ops)
            if y is not None:
                j = basis_map[tuple(y)]
                kappa[j,i] += sign*coeff

        for i,x in enumerate(basis):
            ops = (0,'cre'),(2,'des')
            fill(ops,self.h)
            ops = (0,'cre'),(1,'cre'),(1,'des'),(2,'des')
            fill(ops,-2*self.h)

            ops = (2,'cre'),(0,'des')
            fill(ops,-self.h)
            ops = (2,'cre'),(1,'cre'),(1,'des'),(0,'des')
            fill(ops,2*self.h)
        return kappa
class H2State_GHF(H2State):
    def __init__(self,h,kvec,symmetry='u11'):
        super().__init__(h,kvec,symmetry=symmetry)
        self.nparam -= 1
    def get_x(self,h=None,kvec=None):
        return super().get_x(h=h,kvec=kvec)[1:]
    def update(self,x):
        super()._update(self.h,x)
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

