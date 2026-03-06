import numpy as np
from PyPMG.fermion_state import * 
def get_pairs_u1(nsite):
    return [(p,q) for p in range(nsite-1) for q in range(p)] 
def get_pairs_u11(nsites):
    ls = []
    for i,nsite in enumerate(nsites):
        ls += [(2*p+i,2*q+i) for p in range(nsite-1) for q in range(p)]
        #print([(2*p+i,2*q+i) for p in range(nsite-1) for q in range(p)])
    ls += [(2*p,2*q+1) for p in range(nsites[0]-1) for q in range(nsites[1]-1)]
    #print([(2*p,2*q+1) for p in range(nsites[0]-1) for q in range(nsites[1]-1)])
    return ls
class Jastrow:
    def __init__(self,nsites,symmetry='u11',Jmax=None):
        if symmetry=='u1':
            self.pairs = get_pairs_u1(sum(nsites))
        elif symmetry=='u11':
            self.pairs = get_pairs_u11(nsites)
        else:
            raise ValueError
        #print(self.pairs)
        #exit()
        self.nparam = len(self.pairs)
        self.Jmax = Jmax 
    def _amplitude_and_derivative(self,cf,derivative=True):
        occ = np.array([cf[p]*cf[q] for (p,q) in self.pairs])
        psi_x = np.exp(np.dot(self.x,occ))
        return psi_x,psi_x*occ
    def _amplitude(self,cf):
        return self._amplitude_and_derivative(cf)[0]
    def _update(self,x):
        if self.Jmax is not None:
            for i in range(len(x)):
                if x[i]>self.Jmax:
                    x[i] = self.Jmax
                if x[i]<-self.Jmax:
                    x[i] = -self.Jmax
        self.x = x

class JastrowPMGState(FermionState):
    def __init__(self,jas,pmg):
        super().__init__(pmg.nsites,pmg.nelec,symmetry=pmg.symmetry,thresh=pmg.thresh,rho_swap=pmg.rho_swap,propose_by=pmg.propose_by)
        pmg.get_nparam()
        self.pmg = pmg
        self.jas = jas
        self.nparam = pmg.nparam + jas.nparam
    def _update(self,x):
        self.x = x
        n = self.jas.nparam
        self.jas._update(x[:n])
        self.pmg._update(x[n:])
    def get_x(self):
        return np.concatenate([self.jas.x,self.pmg.x])
    def _amplitude_and_derivative(self,cf,derivative=True):
        psi_1,vx1 = self.jas._amplitude_and_derivative(cf,derivative=derivative)
        psi_2,vx2 = self.pmg._amplitude_and_derivative(cf,derivative=derivative)
        psi_x = psi_1*psi_2
        if not derivative:
            return psi_x,None
        vx = np.concatenate([vx1*psi_2,vx2*psi_1])
        return psi_x,vx
    def _amplitude(self,cf):
        return self._amplitude_and_derivative(cf,derivative=False)[0]
