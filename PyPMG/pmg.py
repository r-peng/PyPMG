import numpy as np
import scipy,itertools,torch
from PyPMG.fermion_state import * 
from PyPMG.utils import *
from PyPMG.vmc import *
np.set_printoptions(linewidth=10000,precision=4)
def get_ctr_ls(decimated,order=1):
    ls = []
    for r in range(1,order+1):
        ls += list(itertools.combinations(decimated,r))
    return ls
def complement(nsite,ls1):
    ls2 = list(set(range(nsite)) - set(ls1))
    ls2.sort()
    return ls2
def get_hop_ls(ls,HF_typ):
    hop_ls = list(itertools.combinations(ls,2))
    if HF_typ=='GHF':
        return hop_ls
    elif HF_typ=='UHF':
        ls = []
        for (p,q) in hop_ls:
            if p%2==q%2:
                ls.append((p,q))
        return ls
    else:
        raise ValueError
def matrix_exp(Y):
    if isinstance(Y,torch.Tensor):
        return torch.linalg.matrix_exp(Y)
    else:
        return scipy.linalg.expm(Y)
def compute_adjugate(A,detA=None,thresh=1e-6):
    n = A.shape[0]
    u,s,v = np.linalg.svd(A)
    if s[0]>thresh:
        if detA is None:
            detA = np.linalg.det(A)
        Ainv = np.dot(v.T.conj()/s.reshape(1,n),u.T.conj())
        return detA*Ainv
    C = np.zeros_like(A)
    for i in range(n):
        rix = list(range(n))
        rix.remove(i)
        Mi = A[np.array(rix)]
        for j in range(n):
            cix = list(range(n))
            cix.remove(j)
            Mij = Mi[:,cix]
            C[i,j] = np.linalg.det(Mij)*(-1)**(i+j)
    return C.T
def determinant_derivative(A,thresh=1e-6):
    detA = np.linalg.det(A)
    adjA = compute_adjugate(A,detA=detA)
    return detA,adjA.T
class PMG:
    def __init__(self,nsite,hop_ls,decimated,order=1,jac_by=None):
        self.nsite = nsite
        self.hop_ls = hop_ls # orbital pair
        self.jac_by = jac_by
        if decimated is None:
            self.decimated = None
            self.ctr_ls = None
            self.nparam = len(self.hop_ls)
        else:
            self.decimated = list(decimated)
            self.decimated.sort()
            self.ctr_ls = get_ctr_ls(self.decimated,order=order)
            self.nparam = len(self.hop_ls)*len(self.ctr_ls)
    def compute_occ(self,cf):
        if self.ctr_ls is None:
            return None
        occ = [None] * len(self.ctr_ls)
        for i,rs in enumerate(self.ctr_ls):
            occ[i] = int(np.prod([cf[r] for r in rs]))
        return tuple(occ)
    def compute_kvec(self,occ,x):
        if occ is None:
            return x
        hnop,nctr = len(self.hop_ls),len(self.ctr_ls)
        x = x.reshape(hnop,nctr) 
        if isinstance(x,torch.Tensor):
            occ_ = torch.tensor(occ,requires_grad=False,dtype=x.dtype)
            return torch.matmul(x,occ_)
        else:
            return np.dot(x,np.array(occ))
    def kvec2Y(self,kvec):
        if isinstance(kvec[0],torch.Tensor):
            Y = torch.zeros(self.nsite,self.nsite,dtype=kvec[0].dtype)
        else:
            Y = np.zeros((self.nsite,)*2,dtype=kvec[0].dtype)
        for val,(p,q) in zip(kvec,self.hop_ls):
            Y[p,q] += val
            Y[q,p] -= val
        return Y
    def compute_rotation(self,cf,x,occ=None):
        if occ is None:
            occ = self.compute_occ(cf)
        kvec = self.compute_kvec(occ,x)
        Y = self.kvec2Y(kvec)
        return matrix_exp(Y) 
    def _update(self):
        pass
class PMG_manual(PMG):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.jac_by=='ad':
            def fxn(kvec):
                Y = self.kvec2Y(kvec)
                U = matrix_exp(Y)
                return U,U
            jac_fxn = torch.func.jacfwd(fxn,has_aux=True)
            def compute_rotation(occ,x):
                kvec = self.compute_kvec(occ,x)
                kvec = torch.tensor(kvec,requires_grad=True)

                jac,U = jac_fxn(kvec)
                U = U.numpy(force=True)
                jac = jac.numpy(force=True)
                kvec.grad = None
                return U,jac
        elif self.jac_by=='frechet':
            E = dict()
            for (p,q) in self.hop_ls:
                Epq = np.zeros((self.nsite,)*2)
                Epq[p,q] = 1
                Epq[q,p] = -1
                E[p,q] = Epq
            def compute_rotation(occ,x):
                kvec = self.compute_kvec(occ,x)
                Y = self.kvec2Y(kvec)

                jac = np.zeros((self.nsite,)*2+(len(self.hop_ls),),dtype=x.dtype)
                for i,(p,q) in enumerate(self.hop_ls):
                    U,jac[:,:,i] = scipy.linalg.expm_frechet(Y,E[p,q])
                return U,jac
        else:
            raise ValueError
        self._compute_rotation = compute_rotation 
    def compute_rotation(self,cf,x,derivative=False):
        occ = self.compute_occ(cf)
        if derivative:
            if occ not in self.ders:
                self.amps[occ],self.ders[occ] = self._compute_rotation(occ,x)
            return self.amps[occ],(self.ders[occ],occ)
        else:
            if occ not in self.amps:
                self.amps[occ] = super().compute_rotation(cf,x,occ=occ)
            return self.amps[occ],None
    def _update(self):
        self.amps = dict()
        self.ders = dict()
class PMGState(FermionState):
    def __init__(self,nsites,nelec,U0=None,**kwargs):
        super().__init__(nsites,nelec,**kwargs)
        # convention: [PMG1]...[(P)MGK]\prod_ic_i^\dagger|vac>
        # orbital rotation: 
        # ...exp(Y_{K-1})exp(Y_{K})(c_i)exp(-Y_{K})exp(-Y_{K-1})...
        # =...exp(Y_{K-1})\sum_{j}c_j[exp(Y_{K})]_{ji}exp(-Y_{K-1})...
        # =...\sum_{jk}c_k[exp(Y_{K-1})]_{kj}[exp(Y_{K})]_{ji}
        self.pmg_ls = []
        self.nparam = None
        self.U0 = U0
    def _process_hop_ls(self,hop_ls,remove_redundant=False,remove_unphysical=None):
        if hop_ls=='GHF':
            hop_ls = list(itertools.combinations(range(self.nsite),2))
        if hop_ls=='UHF':
            hop_ls = []
            for i,nsite in enumerate(self.nsites):
                ls = list(itertools.combinations(range(nsite),2))
                hop_ls += [(2*p+i,2*q+i) for (p,q) in ls]
        if remove_redundant:
            hop_ls = set(hop_ls)
            for pmg in self.pmg_ls:
                hop_ls -= set(pmg.hop_ls)
            hop_ls = list(hop_ls)
        if remove_unphysical is not None:
            hop_ls = set(hop_ls)
            for (p,q) in itertools.combinations(remove_unphysical,2):
                hop_ls.discard((p,q))
            hop_ls = list(hop_ls)
        return hop_ls
    def add_pmg(self,hop_ls,decimated,remove_redundant=False,remove_unphysical=None,**pmg_kwargs):
        hop_ls = self._process_hop_ls(hop_ls,remove_redundant=remove_redundant,remove_unphysical=remove_unphysical)
        pmg = self.pmg_class(self.nsite,hop_ls,decimated,**pmg_kwargs)
        self.pmg_ls.append(pmg)
        if RANK==0:
            print('layer=',len(self.pmg_ls)-1)
            print('nparam=',pmg.nparam)
            print('hop_ls=',pmg.hop_ls)
            print()
    def get_nparam(self):
        self.nparam = sum([pmg.nparam for pmg in self.pmg_ls])
    def _update(self,x):
        self.x = x
        for pmg in self.pmg_ls:
            pmg._update()
    def get_x(self):
        return self.x
    def _amplitude(self,cf):
        return self._amplitude_and_derivative(cf,derivative=False)[0]
class PMGState_autodiff(PMGState):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.U0 is None:
            self.U0_torch = None
        else:
            self.U0 = self.U0[:,:sum(self.nelec)] 
            self.U0_torch = torch.tensor(self.U0,requires_grad=False)
        self.pmg_class = PMG
    def get_mo_coeff(self,cf,derivative=False):
        n = len(self.pmg_ls)
        if derivative:
            dot = torch.matmul 
            U0 = self.U0_torch
            x = torch.tensor(self.x,requires_grad=True) 
        else:
            dot = np.dot
            U0 = self.U0
            x = self.x

        start = 0
        for i,pmg in enumerate(self.pmg_ls):
            stop = start + pmg.nparam
            xi = x[start:stop]
            start = stop

            Ui = pmg.compute_rotation(cf,xi)
            if i==n-1 and U0 is None:
                Ui = Ui[:,:sum(self.nelec)] 
            if i==0:
                U = Ui 
            else:
                U = dot(U,Ui)
        if U0 is not None:
            U = dot(U,U0)
        return x,U
    def _amplitude_and_derivative(self,cf,derivative=True):
        x,U = self.get_mo_coeff(cf,derivative=derivative) 

        idx = np.argwhere(cf).flatten()
        if derivative:
            det = torch.linalg.det
        else:
            det = np.linalg.det
        psi_x = det(U[idx])
        if not derivative:
            return psi_x,None
        psi_x.backward()
        vx = x.grad.numpy(force=True) 
        psi_x = psi_x.numpy(force=True)
        #print(cf,psi_x,vx)
        return psi_x,vx
class PMGState_manual(PMGState):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.U0 is not None:
            self.U0 = self.U0[:,:sum(self.nelec)]
        self.pmg_class = PMG_manual
    def get_mo_coeff(self,cf,derivative=False,lix=None):
        n = len(self.pmg_ls)
        Us = [None] * n + [self.U0]
        Js = [None] * n  
        start = 0
        for i,pmg in enumerate(self.pmg_ls):
            stop = start + pmg.nparam
            xi = self.x[start:stop]
            start = stop

            Us[i],Js[i] = pmg.compute_rotation(cf,xi,derivative=derivative)
            if i==0 and lix is not None:
                Us[i] = Us[i][lix]
        # left sweep:
        lenv = [Us[0]] + [None] * n 
        for i in range(1,n+1):
            Ui = Us[i]
            if Ui is None:
                assert i==n
                lenv[i] = lenv[i-1][:,:sum(self.nelec)]
            else:
                lenv[i] = np.dot(lenv[i-1],Ui)
        return Us,lenv,Js
    def _amplitude_and_derivative(self,cf,derivative=True):
        lix = np.argwhere(cf).flatten()
        Us,lenv,Js = self.get_mo_coeff(cf,derivative=derivative,lix=lix) 
        
        U = lenv[-1]
        if not derivative:
            return np.linalg.det(U),None
        psi_x,dU = determinant_derivative(U) 

        # right sweep:
        n = len(self.pmg_ls)
        renv = [None] * n + [Us[-1]]
        nelec = sum(self.nelec)
        for i in range(n-1,0,-1):
            Ui,prev = Us[i],renv[i+1]
            if prev is None:
                assert i==n-1
                renv[i] = Ui[:,:nelec]
            else:
                renv[i] = np.dot(Ui,prev)
        # jacs
        for i in range(n):
            (Ji,occ),right = Js[i],renv[i+1]
            if right is not None:
                Ji = np.einsum('jkq,kl->jlq',Ji,right)
            else:
                Ji = Ji[:,:nelec]
            if i>0:
                Ji = np.einsum('ij,jkq->ikq',lenv[i-1],Ji)
            else:
                Ji = Ji[lix]
            Ji = np.einsum('ij,ijq->q',dU,Ji)
            if occ is None:
                vi = Ji
            else:
                vi = np.outer(Ji,np.array(occ)).flatten()
            Js[i] = vi 
        vx = np.concatenate(Js)
        #print(cf,psi_x,vx)
        return psi_x,vx
class PMGGraph:
    def __init__(self,backend=None):
        self.backend = backend
        if backend is None:
            self.nodes = dict()
            self.edges = set()
        elif backend=='networkx':
            import networkx as nx
            self.nx = nx
            self.G = self.nx.Graph()
        else:
            raise NotImplementedError
    def has_node(self,name):
        if self.backend is None:
            return (name in self.nodes)
        elif self.backend=='networkx':
            return self.G.has_node(name)
        else:
            raise NotImplementedError
    def parse_node(self,name):
        if self.backend is None:
            hop_ls_left = self.nodes[name]['hop_ls_left']
            ctr_ls_left = self.nodes[name]['ctr_ls_left']
            ctr_ls = self.nodes[name]['ctr_ls']
        elif self.backend=='networkx':
            hop_ls_left = self.G.nodes[name]['hop_ls_left']
            ctr_ls_left = self.G.nodes[name]['ctr_ls_left']
            ctr_ls = self.G.nodes[name]['ctr_ls']
        else:
            raise NotImplementedError
        return ctr_ls,ctr_ls_left,hop_ls_left
    def check_attributes(self,name,ctr_ls_left,hop_ls_left):
        ctr_ls_,ctr_ls_left_,hop_ls_left_ = self.parse_node(name)
        assert ctr_ls_left==ctr_ls_left_
        assert set(hop_ls_left)==set(hop_ls_left_)
    def add_node(self,name,ctr_ls,ctr_ls_left,hop_ls_left):
        if self.has_node(name):
            self.check_attributes(name,ctr_ls_left,hop_ls_left)
            return False
        if self.backend is None:
            node = dict()
            node['hop_ls_left'] = hop_ls_left
            node['ctr_ls_left'] = ctr_ls_left
            node['ctr_ls'] = ctr_ls
            self.nodes[name] = node
        elif self.backend=='networkx':
            self.G.add_node(name)
            self.G.nodes[name]['hop_ls_left'] = hop_ls_left
            self.G.nodes[name]['ctr_ls_left'] = ctr_ls_left
            self.G.nodes[name]['ctr_ls'] = ctr_ls
        else:
            raise NotImplementedError
        return True
    def add_edge(self,n1,n2):
        if self.backend is None:
            self.edges.add((n1,n2))
        elif self.backend=='networkx':
            if not self.G.has_edge(n1,n2):
                self.G.add_edge(n1,n2)
        else:
            raise NotImplementedError
    def get_all_children(self,parent):
        ctr_ls,ctr_ls_left,hop_ls_left = self.parse_node(parent)
        children = []
        for ctr in ctr_ls_left:
            ctr_ls_ = ctr_ls + [ctr]
            ctr_ls_.sort()
            name = tuple(ctr_ls_)
            ctr_ls_left_ = ctr_ls_left.copy()
            ctr_ls_left_.remove(ctr)
            hop_ls_left_ = []
            for (p,q) in hop_ls_left:
                if p==ctr:
                    continue
                if q==ctr:
                    continue
                hop_ls_left_.append((p,q))
            if len(hop_ls_left_)==0:
                continue
            new = self.add_node(name,ctr_ls_,ctr_ls_left_,hop_ls_left_)
            if new:
                children.append(name)
                self.add_edge(parent,name)
        return children
    def get_all_pmg(self,hop_ls,ctr_ls,iprint=0):
        self.add_node('root',ctr_ls=[],ctr_ls_left=ctr_ls,hop_ls_left=hop_ls)
        parents = ['root']
        nix = 0
        layers = [parents]
        while True:
            children = [] 
            for p in parents: 
                children += self.get_all_children(p)
            if len(children)==0:
                break 
            layers.append(children)
            parents = children
            nix += 1
            if iprint>0:
                print(f'\nlayer={nix},number of nodes={len(parents)}')
                for ix,node in enumerate(children):
                    print(f'\tnode {ix}=',node)
                    hop_ls = self.parse_node(node)[-1]
                    print(f'\thop_ls_left=',hop_ls)
        return layers
