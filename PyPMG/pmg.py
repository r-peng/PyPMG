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
class PMG:
    def __init__(self,p,q,nsite,fxn):
        self.p,self.q = p,q # orbital pair
        self.fxn = fxn # fxn(config)

        assert p<q
        self.Y = np.zeros((nsite,)*2)
        self.Y[p,q] = 1 
        self.Y -= self.Y.T
    def _update(self,x):
        self.x = x # param
        self.mo = scipy.linalg.expm(x*self.Y)
    def get_torch_mo(self,cf):
        x = torch.tensor(self.x,requires_grad=True)
        Y = torch.tensor(self.Y,requires_grad=False)

        f = self.fxn(cf)
        return x,torch.linalg.matrix_exp(x*f*Y)
    def get_mo(self,cf):
        f = self.fxn(cf)
        if f==1:
            return self.mo
        if f==-1:
            return self.mo.T
        raise NotImplementedError
    def get_MB_matrix(self,basis):
        basis_map = {b:i for i,b in enumerate(basis)}
        kappa = np.zeros((len(basis),)*2)
        def fill(i,ops,coeff):
            cf = basis[i]
            coeff *= self.fxn(cf) 
            y,sign = string_act(cf,ops)
            if y is not None:
                j = basis_map[tuple(y)]
                kappa[j,i] += sign*coeff

        for i in range(len(basis)):
            ops = (self.p,'cre'),(self.q,'des')
            fill(i,ops,self.x)

            ops = (self.q,'cre'),(self.p,'des')
            fill(i,ops,-self.h)
        return kappa
class MGState(FermionState):
    def __init__(self,nsites,nelec,pairs='GHF',**kwargs):
        super().__init__(nsites,nelec,**kwargs)
        if pairs=='GHF':
            pairs = [(p,q) for p in range(self.nsite) for q in range(p+1,self.nsite)]
        if pairs=='UHF':
            pairs = []
            for i,nsite in enumerate(self.nsites):
                pairs += [(2*p+i,2*q+i) for p in range(nsite) for q in range(p+1,nsite)]
        self.pairs = pairs
        self.Y = dict()
        for (p,q) in self.pairs:
            Y = np.zeros((self.nsite,)*2)
            Y[p,q] = 1
            self.Y[p,q] = Y-Y.T
    def get_x(self):
        return self.x
    def _update(self,x):
        assert len(x)==len(self.pairs)
        self.x = np.array(x) 
        k = 0
        for xi,(p,q) in zip(x,self.pairs):
            k += xi*self.Y[p,q]
        self.mo = scipy.linalg.expm(k)[:,:sum(self.nelec)]
    def get_torch_mo(self):
        x = torch.tensor(self.x,requires_grad=True)
        k = 0
        for xi,(p,q) in zip(x,self.pairs):
            Y = torch.tensor(self.Y[p,q],requires_grad=False)
            k = k+xi*Y
        return x,torch.linalg.matrix_exp(k)[:,:sum(self.nelec)]
    def _amplitude_and_derivative(self,cf):
        raise NotImplementedError
    def _amplitude(self,cf):
        raise NotImplementedError

class PMGState(FermionState):
    def __init__(self,nsites,nelec,pairs='GHF',**kwargs):
        super().__init__(nsites,nelec,**kwargs)
        self.mg = MGState(nsites,nelec,pairs=pairs)
        self.pmg = []
    def add_pmg(self,p,q,fxn):
        # convention: ...(PMG2)(PMG1)|HF>
        # orbital rotation: 
        # ...exp(-Y2)exp(-Y1)(c_i)exp(Y1)exp(Y2)...
        # =...exp(-Y2)\sum_{j}c_j[exp(-Y1)]_{ji}exp(Y2)...
        # =...\sum_{jk}c_k[exp(-Y2)]_{kj}[exp(-Y1)]_{ji}
        pmg = PMG(p,q,self.nsite,fxn)
        self.pmg.append(pmg)
    def _update(self,x):
        n = len(self.mg.pairs)
        assert len(x)==n+len(self.pmg)
        x1,x2 = x[:n],x[n:]
        self.mg._update(x1)
        for i,xi in enumerate(x2):
            self.pmg[i]._update(xi)
    def get_x(self):
        x = np.array([pmg.x for pmg in self.pmg])
        return np.concatenate([self.mg.x,x])
    def _amplitude_and_derivative(self,cf):
        x1,mo = self.mg.get_torch_mo()
        x2 = [None] * len(self.pmg)
        for i,pmg in enumerate(self.pmg):
            x2[i],Ui = pmg.get_torch_mo(cf)
            mo = torch.matmul(Ui,mo)
        idx = np.argwhere(cf).flatten()
        psi_x = torch.linalg.det(mo[idx,:])
        psi_x.backward()
        v1 = x1.grad.numpy(force=True)
        v2 = [xi.grad.numpy(force=True) for xi in x2]
        vx = np.concatenate([v1,v2])
        psi_x = psi_x.numpy(force=True)
        return psi_x,vx
    def _amplitude(self,cf):
        mo = self.mg.mo
        for pmg in self.pmg:
            Ui = pmg.get_mo(cf)
            mo = np.dot(Ui,mo)
        idx = np.argwhere(cf).flatten()
        return np.linalg.det(mo[idx,:])
