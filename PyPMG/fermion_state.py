import numpy as np
import scipy,itertools,time
np.set_printoptions(precision=10,suppress=True)
from PyPMG.hamiltonian import * 
def get_occ_from_mo(mo,nelec):
    ls = []
    for i in range(nelec):
        vec = np.fabs(mo[:,i])
        idx = np.argsort(vec)
        ls.append(idx[-nelec:][::-1])
    ls = np.array(ls)
    occ = set()
    for i in range(nelec):
        lsi = set(ls[:,i])
        lsi -= occ
        n1,n2 = len(occ),len(lsi)
        if n1+n2>nelec:
            lsi = list(lsi)[:nelec-n1]
            occ |= set(lsi)
            return list(occ)
        occ |= lsi
class FermionState:
    def __init__(self,nsites,nelec,**sampling_kwargs):
        self.nsites = nsites
        self.nelec = nelec
        self.nsite = sum(nsites)
        self.pc = precompute_for_exc_ls(self.nsite)
        self.set_sampling_kwargs(**sampling_kwargs)
    def set_sampling_kwargs(self,symmetry='u11',thresh=1e-10,rho_swap=0.,propose_by='uniform'):
        self.symmetry = symmetry
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
    def get_random_config(self,rng,occ=None):
        if self.symmetry=='u1':
            return get_random_config_u1(self.nsite,sum(self.nelec),rng,occ=occ)
        elif self.symmetry=='u11':
            return get_random_config_u11(self.nsites,self.nelec,rng,occ=occ) 
        #elif self.symmetry=='fock':
        #    return list(itertools.product((0,1),repeat=self.nsite))
        else:
            raise NotImplementedError
    def _propose_uniform(self,x):
        t0 = time.time()
        if self.symmetry=='u1':
            ls = get_exc_list_u1(x,self.nsite)
        elif self.symmetry=='u11':
            ls = get_exc_list_u11(x,self.pc)
        else:
            raise NotImplementedError
        q = 1./len(ls)
        print('propose uniform time=',time.time()-t0)
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
            print('no_swap')
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
        correct_symmetry = check_symmetry(x,self.symmetry,self.nsite,self.nelec)
        if not correct_symmetry:
            return 0
        if x in self.amps:
            return self.amps[x]
        self.amps[x] = self._amplitude(x)
        return self.amps[x] 
    def log_prob(self,x):
        correct_symmetry = check_symmetry(x,self.symmetry,self.nsite,self.nelec)
        if not correct_symmetry:
            return None 
        psi_x = self.amplitude(x)
        return np.log(psi_x*psi_x.conj())
    def amplitude_and_derivative(self,x):
        correct_symmetry = check_symmetry(x,self.symmetry,self.nsite,self.nelec)
        if not correct_symmetry:
            return 0,None 
        if x in self.ders:
            return self.amps[x],self.ders[x]
        self.amps[x],vx = self._amplitude_and_derivative(x)
        self.ders[x] = vx
        return self.amps[x],self.ders[x]
