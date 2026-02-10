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
def get_MB_Sz(nsite,basis):
    basis_map = {b:i for i,b in enumerate(basis)}
    M = np.zeros((len(basis),)*2)
    for i,x in enumerate(basis):
        Mii = 0
        for p in range(nsite):
            Mii += x[p]*(-1)**(p%2)
        M[i,i] = Mii/2
    return M
def get_MB_Spm(nsite,basis,typ):
    basis_map = {b:i for i,b in enumerate(basis)}
    M = np.zeros((len(basis),)*2)
    for i,x in enumerate(basis):
        for p in range(nsite//2):
            if typ=='+':
                ops = (2*p,'cre'),(2*p+1,'des')
            else:
                ops = (2*p+1,'cre'),(2*p,'des')
            y,sign = string_act(x,ops)
            if y is None:
                continue
            j = basis_map[y]
            M[j,i] += sign
    return M
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
class Operator:
    def __init__(self,thresh=1e-6,weight=1):
        self.thresh = thresh 
        self.weight = weight 
        self.elocs = dict()
        self.hs = dict()
    def add_term(self,x,ops,coeff):
        if np.fabs(coeff)<self.thresh:
            return 
        y,sign = string_act(x,ops)
        if y is None:
            return 
        if y not in self.terms:
            self.terms[y] = 0
        self.terms[y] += sign*coeff
    def compute_eloc(self,x,psi,compute_h=False):
        if x in self.elocs:
            return 
        psi_x = psi.amplitude(x)
        if np.fabs(psi_x)<self.thresh:
            self.elocs[x] = 0
            if compute_h:
                self.hs[x] = np.zeros(psi.nparam)
            return
        terms = self.eloc_terms(x)
        eloc = 0 
        hx = 0 
        for (y,coeff) in terms.items():
            if compute_h:
                psi_y,vy = psi.amplitude_and_derivative(y)
                hx += vy*coeff
            else:
                psi_y = psi.amplitude(y)
            eloc += psi_y*coeff
        self.elocs[x] = eloc/psi_x
        if compute_h:
            self.hs[x] = hx/psi_x
    def get_MB_matrix(self,basis=None,nelec=None,symmetry='u1'):
        if basis is None:
            if symmetry=='u1':
                basis = get_all_configs_u1(self.nao*2,nelec)
            elif symmetry=='u11':
                basis = get_all_configs_u11((self.nao,)*2,nelec)
            else:
                raise NotImplementedError
        basis_map = {b:i for i,b in enumerate(basis)}
        M = np.zeros((len(basis),)*2)
        for i,x in enumerate(basis):
            terms = self.eloc_terms(x)
            for y,coeff in terms.items():
                j = basis_map[y]
                M[j,i] += coeff
        return M,basis
class QCHamiltonian(Operator):
    def __init__(self,hcore,eri,thresh=1e-6):
        super().__init__(thresh=thresh)
        self.nao = hcore.shape[0]
        self.hcore = hcore # ao-integrals
        self.eri_qc = eri.copy()

        eri = eri.transpose(0,2,1,3) # permute to physicist notation (b1,b2,k1,k2)
        nso = self.nao*2
        v = np.zeros((nso,nso,nso,nso))
        v[::2,::2,::2,::2] = eri.copy()
        v[1::2,1::2,1::2,1::2] = eri.copy()
        v[::2,1::2,::2,1::2] = eri.copy()
        v[1::2,::2,1::2,::2] = eri.copy()
        self.eri = v-v.transpose(0,1,3,2)
        self.eri /= 4
    def eloc_terms(self,x):
        self.terms = {}
        for i in (0,1):
            for (p,q) in itertools.product(range(self.nao),repeat=2):
                ops = (2*p+i,'cre'),(2*q+i,'des')
                self.add_term(x,ops,self.hcore[p,q])
        for (p,q,r,s) in itertools.product(range(self.nao*2),repeat=4):
            ops = (p,'cre'),(q,'cre'),(s,'des'),(r,'des')
            self.add_term(x,ops,self.eri[p,q,r,s])
        return self.terms
class TotalSpin(Operator):
    def __init__(self,nao,weight=1):
        super().__init__(weight=weight)
        self.nao = nao
    def eloc_terms(self,x):
        xup,xdown = x[::2],x[1::2]
        self.terms = {}
        # Sz**2
        self.terms[x] = (sum(xup)-sum(xdown))**2/4.
        # SpSm+SmSp
        self.terms[x] += sum(x)/2.
        for i in range(self.nao):
            self.terms[x] -= x[2*i]*x[2*i+1]
        for i in range(self.nao):
            for j in range(i+1,self.nao):
                ops = (2*i,'cre'),(2*j+1,'cre'),(2*j,'des'),(2*i+1,'des')
                self.add_term(x,ops,1)
                ops = (2*i+1,'cre'),(2*j,'cre'),(2*j+1,'des'),(2*i,'des')
                self.add_term(x,ops,1)
        return self.terms
class MBOperator(Operator):
    def __init__(self,matrix,basis,thresh=1e-6):
        self.matrix = matrix
        self.basis = basis
        self.basis_map = {b:i for i,b in enumerate(basis)}
        self.thresh = thresh 
        self.elocs = dict()
        self.weight = 1.
    def eloc_terms(self,x):
        terms = {}
        xix = self.basis_map[x]
        for y,val in zip(self.basis,self.H[xix]): 
            if np.absolute(val)>self.thresh:
                terms[y] = val
        return terms

