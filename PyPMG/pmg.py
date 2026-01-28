import numpy as np
import scipy,itertools
import torch
from PyPMG.fermion_state import * 
class PMG:
    def __init__(self,h,p,q,nsite,fxn):
        self.p,self.q = p,q # orbital pair
        self.fxn # fxn(config)

        assert p<q
        self.hmat = np.zeros((nsite,nsite))
        self.hmat[p,q] = 1 
        self.hmat -= self.hmat.T
        self._update(h)
    def _update(self,h):
        self.h = h 
        self.left = scipy.linalg.expm(-self.h*self.hmat)
    def torch_mo(self,x):
        h = torch.tensor(self.h,requires_grad=True)
        hmat = torch.tensor(self.hmat,requires_grad=False)

        f = self.fxn(x)
        left = torch.linalg.matrix_exp(-h*f*hmat)
        idx = np.argwhere(x).flatten()
        return left[:,idx]
    def mo(self,x):
        f = self.fxn(x)
        if f==1:
            return self.left
        if f==-1:
            return self.left.T
        raise NotImplementedError
