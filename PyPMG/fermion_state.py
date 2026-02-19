import numpy as np
import scipy,itertools
np.set_printoptions(precision=10,suppress=True)
from PyPMG.hamiltonian import * 
def get_exc_list(x,nexs=2,symmetry='u11'):
    na,nb = sum(x[::2]),sum(x[1::2])
    xarr = np.array(x) 
    occ = np.argwhere(xarr>0.5).flatten()
    vir = np.argwhere(xarr<0.5).flatten()
    new_cfs = []
    for nex in range(1,nexs+1):
        occ_n = list(itertools.combinations(occ,nex))
        vir_n = list(itertools.combinations(vir,nex))
        for oix,vix in itertools.product(occ_n,vir_n): 
            y = list(x)
            for i,a in zip(oix,vix):
                y[i] = 1-y[i]
                y[a] = 1-y[a]
            if symmetry=='u11':
                if sum(y[::2])!=na:
                    continue
                if sum(y[1::2])!=nb:
                    continue
            new_cfs.append(tuple(y))
    return new_cfs
def get_swap_list(x):
    xa,xb = x[::2],x[1::2]
    xa_arr,xb_arr = np.array(xa),np.array(xb) 
    u = np.argwhere(xa_arr*(1-xb_arr)>0.5).flatten() # singly occ up
    v = np.argwhere(xb_arr*(1-xa_arr)>0.5).flatten() # singly occ down
    new_cfs = []
    for p,q in itertools.product(u,v):
        y = list(x)
        for i in (2*p,2*q+1,2*q,2*p+1):
            y[i] = 1-y[i]
        new_cfs.append(tuple(y))
    return new_cfs 
class FermionState:
    def __init__(self,nsites,nelec,symmetry='u11',thresh=1e-10,rho_swap=0.,propose_by='uniform'):
        self.nsites = nsites
        self.nelec = nelec
        self.nsite = sum(nsites)
        self.symmetry = symmetry
        self.amps = dict()
        self.ders = dict()
        self.thresh = thresh # |amplitude|<thresh are treated as 0
        self.rho_swap = rho_swap
        self.propose_by = propose_by 
    def update(self,x):
        self._update(x)
        self.amps = dict()
        self.ders = dict()
    def get_all_configs(self):
        if self.symmetry=='u1':
            return get_all_configs_u1(self.nsite,sum(self.nelec))
        elif self.symmetry=='u11':
            return get_all_configs_u11(self.nsites,self.nelec) 
        elif self.symmetry=='fock':
            return list(itertools.product((0,1),repeat=self.nsite))
        else:
            raise NotImplementedError
    def _propose_uniform(self,x):
        ls = get_exc_list(x,symmetry=self.symmetry)
        q = 1./len(ls)
        return {cf:q for cf in ls}
    def _propose_ham(self,x,ham):
        cfs = ham.eloc_terms(x)
        cfs.pop(x)
        n = np.absolute(np.array(list(cfs.values()))).sum()
        return {cf:np.absolute(val)/n for cf,val in cfs.items()}
    def _propose(self,x,ham=None):
        if self.propose_by=='uniform':
            cfs = self._propose_uniform(x)
        elif self.propose_by=='hamiltonian':
            cfs = self._propose_ham(x,ham)
        else:
            raise ValueError(f'self.propose_by={self.propose_by} not implemented.')
        if self.rho_swap<self.thresh:
            return cfs
        ls = get_swap_list(x)
        if len(ls)==0:
            return cfs
        fac = 1.-self.rho_swap
        cfs = {cf:val*fac for cf,val in cfs.items()}
        q = self.rho_swap/len(ls)
        for cf in ls:
            if cf not in cfs:
                cfs[cf] = 0
            cfs[cf] += q
        return cfs
    def propose(self,x,rng,ham=None):
        cfs = self._propose(x,ham=ham)
        keys = list(cfs.keys())
        p = [cfs[cf] for cf in keys]
        #print(f'p={p},sum={sum(p)}')
        ix = rng.choice(len(keys),p=p)
        return keys[ix],p[ix] 
    def propose_reverse(self,x,y,ham=None):
        cfs = self._propose(y,ham=ham)
        return cfs[x]
    def amplitude(self,x):
        if self.symmetry=='u1':
            if sum(x)!=sum(self.nelec):
                return 0
        if self.symmetry=='u11':
            if sum(x[::2])!=self.nelec[0]:
                return 0
            if sum(x[1::2])!=self.nelec[1]:
                return 0
        if x in self.amps:
            return self.amps[x]
        self.amps[x] = self._amplitude(x)
        return self.amps[x] 
    def log_prob(self,x):
        if self.symmetry=='u1':
            if sum(x)!=sum(self.nelec):
                return None
        if self.symmetry=='u11':
            if sum(x[::2])!=self.nelec[0]:
                return None
            if sum(x[1::2])!=self.nelec[1]:
                return None
        psi_x = self.amplitude(x)
        return np.log(psi_x*psi_x.conj())
    def amplitude_and_derivative(self,x):
        if self.symmetry=='u1':
            if sum(x)!=sum(self.nelec):
                return 0,None
        if self.symmetry=='u11':
            if sum(x[::2])!=self.nelec[0]:
                return 0,None
            if sum(x[1::2])!=self.nelec[1]:
                return 0,None
        if x in self.ders:
            return self.amps[x],self.ders[x]
        self.amps[x],vx = self._amplitude_and_derivative(x)
        self.ders[x] = vx
        return self.amps[x],self.ders[x]
