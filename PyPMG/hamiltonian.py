import numpy as np
import scipy,itertools,time
#from PyPMG.walker_numpy_occvec import *
from PyPMG.walker_bitstring import *
np.set_printoptions(precision=10,suppress=True)
# TODO: make compatible with bitstring
#def get_MB_Sz(nsite,basis):
#    basis_map = {b:i for i,b in enumerate(basis)}
#    M = np.zeros((len(basis),)*2)
#    for i,x in enumerate(basis):
#        Mii = 0
#        for p in range(nsite):
#            Mii += x[p]*(-1)**(p%2)
#        M[i,i] = Mii/2
#    return M
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
class Operator:
    def __init__(self,thresh=1e-6,weight=1,save_link=True):
        self.thresh = thresh 
        self.weight = weight 
        self.save_link = save_link
        self.elocs = dict()
        self.hs = dict()
        self.links = dict() # <x|H|y>
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
        cfs,coeffs = self.eloc_terms(x)
        eloc = 0 
        hx = 0 
        for (y,coeff) in zip(cfs,coeffs):
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
    def _eloc_terms(self,x):
        if x in self.links:
            return self.links[x]
        t0 = time.time()
        self.terms = {}
        for i in (0,1):
            for (p,q) in itertools.product(range(self.nao),repeat=2):
                ops = (2*p+i,'cre'),(2*q+i,'des')
                self.add_term(x,ops,self.hcore[p,q])
        for (p,q,r,s) in itertools.product(range(self.nao*2),repeat=4):
            ops = (p,'cre'),(q,'cre'),(s,'des'),(r,'des')
            self.add_term(x,ops,self.eri[p,q,r,s])

        cfs = []
        coeffs = []
        for y,coeff in self.terms.items():
            cfs.append(y)
            coeffs.append(coeff)
        self.terms = None
        coeffs = np.array(coeffs) 
        if self.save_link:
            self.links[x] = cfs,coeffs
        print('eloc terms time=',time.time()-t0)
        return cfs,coeffs
    def eloc_terms(self,x):
        if x in self.links:
            return self.links[x]
        occ = np.array(get_occ_indices(x))
        occ_a = occ[occ%2==0]//2 
        occ_b = (occ[occ%2==1]-1)//2
        vir = np.array(get_vir_indices(x,self.nao*2))
        vir_a = vir[vir%2==0]//2
        vir_b = (vir[vir%2==1]-1)//2
        print(occ,occ_a,occ_b)
        print(vir,vir_a,vir_b)
        exit()


        cfs = [x]
        # diagonal E1
        hcore_diag = np.diag(self.hcore)
        E1 = sum(hcore_diag[occ_a]) + sum(hcore_diag[occ_b])
        # diagonal E2
        eri = self.eri_qc 
        eri_xxoo = eri[:,:,:,occ_a][:,:,occ_a]
        eri_oooo = eri_xxoo[occ_a][:,occ_a]
        E2 = .5*np.einsum('iijj->',eri_oooo)
        E2 -= .5*np.einsum('ijji->',eri_oooo)
        eri_xxOO = eri[:,:,:,occ_b][:,:,occ_b]
        eri_OOOO = eri_xxOO[occ_b][:,occ_b]
        E2 += .5*np.einsum('iijj->',eri_OOOO)
        E2 -= .5*np.einsum('ijji->',eri_OOOO)
        eri_ooOO = eri_xxOO[occ_a][:,occ_a]
        E2 += np.einsum('iijj->',eri_ooOO)
        coeff = [E1+E2] 
        eri_oooo = eri_OOOO = eri_ooOO = None

        # singles alpha
        F = self.hcore[vir_a][:,occ_a]
        eri_vooo = eri_xxoo[vir_a][:,occ_a]
        F += np.einsum('aijj->ai',eri_vooo)
        F -= np.einsum('ajji->ai',eri_vooo)
        eri_voOO = eri_xxOO[vir_a][:,occ_a]
        F += np.einsum('aijj->ai',eri_voOO)
        for ix_i,i in enumerate(occ_a):
            for ix_a,a in enumerate(vir_a):
                y,s = config_map_single(x,2*i,2*a)
                cfs.append(y)
                coeffs.append(s*F[ix_a,ix_i])
        eri_vooo = eri_voOO = None
        # singles beta 
        F = self.hcore[vir_b][:,occ_b]
        eri_VOOO = eri_xxOO[vir_b][:,occ_b]
        F += np.einsum('aijj->ai',eri_VOOO)
        F -= np.einsum('ajji->ai',eri_VOOO)
        eri_VOoo = eri_xxoo[vir_b][:,occ_b]
        F += np.einsum('aijj->ai',eri_VOoo)
        for ix_i,i in enumerate(occ_b):
            for ix_a,a in enumerate(vir_b):
                y,s = config_map_single(x,2*i+1,2*a+1)
                cfs.append(y)
                coeffs.append(s*F[ix_a,ix_i])
        eri_VOOO = eri_VOoo = None
        eri_xxoo = eri_xxOO = None

        # doubles aa
        for i,j in itertools.combinations(occ_a,2):
            for a,b in itertools.combinations(vir_a,2):
                coeff = eri[a,i,b,j]-eri[a,j,b,i]
                y,s = config_map_doubles(x,2*i,2*j,2*a,2*b)
                cfs.append(y)
                coeffs.append(s*coeff)
        # doubles bb 
        for i,j in itertools.combinations(occ_b,2):
            for a,b in itertools.combinations(vir_b,2):
                coeff = eri[a,i,b,j]-eri[a,j,b,i]
                y,s = config_map_doubles(x,2*i+1,2*j+1,2*a+1,2*b+1)
                cfs.append(y)
                coeffs.append(s*coeff)
        # doubles ab
        for i,j in itertools.product(occ_a,occ_b):
            for a,b in itertools.product(vir_a,vir_b):
                coeff = eri[a,i,b,j]
                y,s = config_map_doubles(x,2*i,2*j+1,2*a,2*b+1)
                cfs.append(y)
                coeffs.append(s*coeff)
# TODO: make compatible with bitstring
#class TotalSpin(Operator):
#    def __init__(self,nao,weight=1):
#        super().__init__(weight=weight)
#        self.nao = nao
#    def eloc_terms(self,x):
#        xup,xdown = x[::2],x[1::2]
#        self.terms = {}
#        # Sz**2
#        self.terms[x] = (sum(xup)-sum(xdown))**2/4.
#        # SpSm+SmSp
#        self.terms[x] += sum(x)/2.
#        for i in range(self.nao):
#            self.terms[x] -= x[2*i]*x[2*i+1]
#        for i in range(self.nao):
#            for j in range(i+1,self.nao):
#                ops = (2*i,'cre'),(2*j+1,'cre'),(2*j,'des'),(2*i+1,'des')
#                self.add_term(x,ops,1)
#                ops = (2*i+1,'cre'),(2*j,'cre'),(2*j+1,'des'),(2*i,'des')
#                self.add_term(x,ops,1)
#        return self.terms
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

