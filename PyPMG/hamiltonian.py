import numpy as np
import scipy,itertools
np.set_printoptions(precision=10,suppress=True)
def fermion_act(x,oix,typ):
    kill = {'cre':1,'des':0}[typ]
    if x[oix]==kill:
        return None,0
    sign = (-1)**sum(x[:oix])
    y = list(x)
    y[oix] = 1-x[oix] 
    return tuple(y),sign
def string_act(x,ops,order=-1):
    if order==-1:
        ops = ops[::-1]
    s = np.zeros(len(ops),dtype=int)
    y = list(x) 
    for i,(oix,typ) in enumerate(ops):
        y,s[i] = fermion_act(y,oix,typ)
        if y is None:
            return None,0
    return tuple(y),np.prod(s)
def get_all_configs_u1(nsites,nelecs):
    configs = []
    for cf in itertools.product((0,1),repeat=nsites):
        if sum(cf)!=nelecs:
            continue
        configs.append(tuple(cf))
    return configs
def get_all_configs_u11(nsites,nelecs):
    alpha = get_all_configs_u1(nsites[0],nelecs[0])
    beta = get_all_configs_u1(nsites[1],nelecs[1])
    configs = []
    for cfa,cfb in itertools.product(alpha,beta):
        cf = [None] * (len(cfa)+len(cfb))
        cf[::2] = cfa
        cf[1::2] = cfb
        configs.append(tuple(cf))
    return configs
def new_configs(x,nexs=2,symmetry='u11'):
    x = list(x)
    na,nb = sum(x[::2]),sum(x[1::2])
    occ = np.argwhere(np.array(x)>0.5).flatten()
    vir = np.argwhere(np.array(x)<0.5).flatten()
    new_cfs = []
    for nex in range(1,nexs+1):
        occ_ = list(itertools.combinations(occ,nex))
        vir_ = list(itertools.combinations(vir,nex))
        for oix,vix in itertools.product(occ_,vir_): 
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
class QCHamiltonian:
    def __init__(self,hcore,eri):
        self.nao = hcore.shape[0]
        self.hcore = hcore # ao-integrals

        eri = eri.transpose(0,2,1,3) # permute to physicist notation (b1,b2,k1,k2)
        nso = self.nao*2
        v = np.zeros((nso,nso,nso,nso))
        v[::2,::2,::2,::2] = eri.copy()
        v[1::2,1::2,1::2,1::2] = eri.copy()
        v[::2,1::2,::2,1::2] = eri.copy()
        v[1::2,::2,1::2,::2] = eri.copy()
        self.eri = v-v.transpose(0,1,3,2)
        self.eri /= 4

        self.thresh = 1e-6
    def eloc_terms(self,x):
        terms = {}
        for i in (0,1):
            for (p,q) in itertools.product(range(self.nao),repeat=2):
                ops = (2*p+i,'cre'),(2*q+i,'des')
                y,sign = string_act(x,ops)
                if y is None:
                    continue
                coeff = sign*self.hcore[p,q]
                if np.fabs(coeff)>self.thresh:
                    if y not in terms:
                        terms[y] = 0
                    terms[y] += coeff

        for (p,q,r,s) in itertools.product(range(self.nao*2),repeat=4):
            ops = (p,'cre'),(q,'cre'),(s,'des'),(r,'des')
            y,sign = string_act(x,ops)
            if y is None:
                continue
            y = tuple(y)
            coeff = sign*self.eri[p,q,r,s]
            if np.fabs(coeff)>self.thresh:
                if np.fabs(coeff)>self.thresh:
                    if y not in terms:
                        terms[y] = 0
                    terms[y] += coeff
        return terms
    def get_MB_hamiltonian(self,basis=None,nelec=None,symmetry='u1'):
        if basis is None:
            if symmetry=='u1':
                basis = get_all_configs_u1(self.nao*2,nelec)
            elif symmetry=='u11':
                basis = get_all_configs_u11((self.nao,)*2,nelec)
            else:
                raise NotImplementedError
        basis_map = {b:i for i,b in enumerate(basis)}
        H = np.zeros((len(basis),)*2)
        for i,x in enumerate(basis):
            terms = self.eloc_terms(x)
            for y,coeff in terms.items():
                j = basis_map[y]
                H[j,i] += coeff
        return H,basis
class MBHamiltonian:
    def __init__(self,H,basis):
        self.H = H
        self.basis = basis
        self.basis_map = {b:i for i,b in enumerate(basis)}
        self.thresh = 1e-6
    def eloc_terms(self,x):
        terms = {}
        xix = self.basis_map[x]
        for y,val in zip(self.basis,self.H[xix]): 
            if np.absolute(val)>self.thresh:
                terms[y] = val
        return terms
class FermionState:
    def __init__(self,nsite,nelec,symmetry='u11'):
        self.nsite = nsite
        self.nelec = nelec
        self.symmetry = symmetry
        self.nnew = None
    def get_all_configs(self):
        if self.symmetry=='u1':
            return get_all_configs_u1(self.nsite,sum(self.nelec))
        elif self.symmetry=='u11':
            return get_all_configs_u11((self.nsite//2,)*2,self.nelec) 
        else:
            raise NotImplementedError
    def new_configs(self,x):
        ls = new_configs(x,symmetry=self.symmetry)
        if self.nnew is None:
            self.nnew = len(ls)
        else:
            if len(ls)!=self.nnew:
                print(len(ls),self.nnew)
                exit()
        return ls
    def log_prob(self,x):
        if self.symmetry=='u1':
            if sum(x)!=sum(self.nelec):
                return 0,None
        if self.symmetry=='u11':
            if sum(x[::2])!=self.nelec[0]:
                return 0,None
            if sum(x[1::2])!=self.nelec[1]:
                return 0,None

        psi_x = self.amplitude(x)
        return np.log(psi_x*psi_x.conj())
