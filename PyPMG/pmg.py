import numpy as np
import scipy,itertools
import torch
from PyPMG.fermion_state import * 
def get_MB_matrix(basis,hop_ls,x,fxn):
    basis_map = {b:i for i,b in enumerate(basis)}
    Y = np.zeros((len(basis),)*2)
    def fill(i,ops,coeff):
        cf = basis[i]
        coeff *= fxn(cf) 
        y,sign = string_act(cf,ops)
        if y is not None:
            j = basis_map[tuple(y)]
            Y[j,i] += sign*coeff

    for i in range(len(basis)):
        for xi,(p,q) in zip(x,hop_ls):
            ops = (p,'cre'),(q,'des')
            fill(i,ops,xi)

            ops = (q,'cre'),(p,'des')
            fill(i,ops,-xi)
    return Y 
def _prodZ(cf,orbs):
    facs = [1-2*cf[i] for i in orbs]
    return np.prod(facs)
def _default_ctr_fxn(cf,x,ctr_ls):
    facs = [_prodZ(cf,key) for key in ctr_ls] 
    return facs[0]+sum([xi*fi for xi,fi in zip(x,facs[1:])])
class PMG:
    def __init__(self,nsite,hop_ls,fxn=None,ctr_ls=None):
        self.nsite = nsite
        self.hop_ls = hop_ls # orbital pair
        self.Y = dict()
        for (p,q) in self.hop_ls:
            Y = np.zeros((nsite,)*2)
            Y[p,q] = 1
            self.Y[p,q] = torch.tensor(Y-Y.T,requires_grad=False)

        self.nparam = len(hop_ls) 
        if ctr_ls is not None:
            self.ctr_ls = ctr_ls
            def fxn(cf,x):
                return _default_ctr_fxn(cf,x,self.ctr_ls)
            self.nparam += len(ctr_ls)-1
        self.fxn = fxn
    def _update(self,x):
        self.x = np.array(x) # param

        Y = np.zeros((self.nsite,)*2)
        for xi,(p,q) in zip(x,self.hop_ls):
            Y[p,q] = xi
            Y[q,p] = -xi
        self.Y['full'] = Y
        self.Y['expY'] = scipy.linalg.expm(self.Y['full'])
    def get_mo_derivative(self,cf):
        x = torch.tensor(self.x,requires_grad=True)
        Y = sum([xi*self.Y[p,q] for xi,(p,q) in zip(x,self.hop_ls)])

        n = len(self.hop_ls)
        f = self.fxn(cf,x[n:])
        return x,torch.linalg.matrix_exp(f*Y)
    def get_mo(self,cf,derivative=False):
        if derivative:
            return self.get_mo_derivative(cf)
        n = len(self.hop_ls)
        f = self.fxn(cf,self.x[n:])
        if f==1:
            return None,self.Y['expY']
        if f==-1:
            return None,self.Y['expY'].T
        return None,scipy.linalg.expm(f*self.Y['full'])
    def get_MB_matrix(self,basis):
        n = len(self.hop_ls)
        def fxn(cf):
            return self.fxn(cf,self.x[n:])
        return get_MB_matrix(basis,self.hop_ls,self.x[:n],fxn)
class MG(PMG):
    def __init__(self,nsites,hop_ls='GHF'):
        if hop_ls=='GHF':
            nsite = sum(nsites)
            hop_ls = [(p,q) for p in range(nsite) for q in range(p+1,nsite)]
        if hop_ls=='UHF':
            hop_ls = []
            for i,nsite in enumerate(nsites):
                hop_ls += [(2*p+i,2*q+i) for p in range(nsite) for q in range(p+1,nsite)]
        def fxn(cf,x):
            return 1
        super().__init__(sum(nsites),hop_ls,fxn)
class RMG(PMG):
    def __init__(self,nsites):
        na = nsites[0]
        hop_ls = [(2*p,2*q) for p in range(na) for q in range(p+1,na)]
        def fxn(cf,x):
            return 1
        super().__init__(sum(nsites),hop_ls,fxn)
        for (p,q) in self.Y:
            self.Y[p,q][p+1,q+1] = 1
            self.Y[p,q][q+1,p+1] = -1
    def _update(self,x):
        self.x = np.array(x) # param

        Y = np.zeros((self.nsite,)*2)
        for xi,(p,q) in zip(x,self.hop_ls):
            Y[p,q] = Y[p+1,q+1] = xi
            Y[q,p] = Y[q+1,p+1] = -xi
        self.Y['full'] = Y
        self.Y['expY'] = scipy.linalg.expm(self.Y['full'])
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
    def _amplitude_and_derivative(self,cf):
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
class PMGState(FermionState):
    def __init__(self,nsites,nelec,pmg_ls,U0=None,**kwargs):
        super().__init__(nsites,nelec,**kwargs)
        # convention: ...[PMG2][(P)MG1]\prod_ic_i^\dagger|vac>
        # orbital rotation: 
        # ...exp(Y2)exp(Y1)(c_i)exp(-Y1)exp(-Y2)...
        # =...exp(Y2)\sum_{j}c_j[exp(Y1)]_{ji}exp(-Y2)...
        # =...\sum_{jk}c_k[exp(Y2)]_{kj}[exp(Y1)]_{ji}
        self.pmg_ls = pmg_ls 
        self.nparam = sum([pmg.nparam for pmg in self.pmg_ls])
        if U0 is None:
            self.U0 = self.U0_torch = None
        else:
            self.U0 = U0[:,:sum(self.nelec)] 
            self.U0_torch = torch.tensor(self.U0,requires_grad=False)
    def _update(self,x):
        for pmg in self.pmg_ls:
            xi,x = x[:pmg.nparam],x[pmg.nparam:]
            pmg._update(xi)
    def get_x(self):
        x = [pmg.x for pmg in self.pmg_ls]
        return np.concatenate(x)
    def _amplitude_and_derivative(self,cf,derivative=True):
        n = len(self.pmg_ls)
        x = [None] * n 
        idx = np.argwhere(cf).flatten()
        if derivative:
            dot = torch.matmul 
            det = torch.linalg.det
            U0 = self.U0_torch
        else:
            dot = np.dot
            det = np.linalg.det
            U0 = self.U0
        for i,pmg in enumerate(self.pmg_ls):
            x[i],Ui = pmg.get_mo(cf,derivative=derivative)
            if i==n-1 and U0 is None:
                Ui = Ui[:,:sum(self.nelec)] 
            if i==0:
                U = Ui[idx,:] 
            else:
                U = dot(U,Ui)
        if U0 is not None:
            U = dot(U,U0)
        psi_x = det(U)
        if not derivative:
            return psi_x,None
        psi_x.backward()
        vx = [xi.grad.numpy(force=True) for xi in x]
        psi_x = psi_x.numpy(force=True)
        return psi_x,np.concatenate(vx)
    def _amplitude(self,cf):
        return self._amplitude_and_derivative(cf,derivative=False)[0]
class JastrowPMGState(PMGState):
    def __init__(self,nsites,nelec,pmg_ls,jastrow_ls,U0=None,**kwargs):
        super().__init__(nsites,nelec,pmg_ls,U0=U0,**kwargs)
        self.jastrow_ls = jastrow_ls
        for jas in jastrow_ls:
            self.nparam += jas.nparam
    def _update(self,x):
        for jas in self.jastrow_ls:
            xi,x = x[:jas.nparam],x[jas.nparam:]
            jas._update(xi)
        super()._update(x)
    def get_x(self):
        x = [jas.x for jas in self.jastrow_ls]
        return np.concatenate(x+[super().get_x()])
    def _amplitude_and_derivative(self,cf,derivative=True):
        psi_x = [None] * (len(self.jastrow_ls)+1)
        vx = [None] * (len(self.jastrow_ls)+1)
        psi_x[-1],vx[-1] = super()._amplitude_and_derivative(cf,derivative=derivative) 
        for i,jas in enumerate(self.jastrow_ls):
            psi_x[i],vx[i] = jas._amplitude_and_derivative(cf)
            vx[i] *= psi_x[-1]
        psi_x = np.prod(psi_x)
        if derivative:
            return psi_x,np.concatenate(vx)
        else:
            return psi_x,None
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
