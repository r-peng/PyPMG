import numpy as np
from PyPMG.fermion_state import * 
class Jastrow:
    def __init__(self,gps=None,nsite=None,Jmax=None):
        if gps is None: 
            self.gps = None
            self.pairs = [(p,q) for p in range(nsite) for q in range(p+1,nsite)] 
            self.nparam = len(self.pairs)
        else:
            self.pairs = None
            self.gps = gps 
            self.nparam = len(self.gps)
        self.Jmax = Jmax 
    def _amplitude_and_derivative(self,cf,derivative=True):
        if self.gps is not None:
            occ = np.zeros(len(self.gps))
            for i,gp in enumerate(self.gps):
                occ[i] = sum([cf[p]*cf[q] for (p,q) in gp])
        if self.pairs is not None:
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
