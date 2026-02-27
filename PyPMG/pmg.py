import numpy as np
import scipy,itertools,torch
from PyPMG.fermion_state import * 
from PyPMG.utils import *
from PyPMG.vmc import *
np.set_printoptions(linewidth=10000,precision=4)
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
def get_ctr_ls(decimated,order=1):
    ls = []
    for r in range(1,order+1):
        ls += list(itertools.combinations(decimated,r))
    return ls
def _prodZ(cf,orbs):
    facs = [1-2*cf[i] for i in orbs]
    return np.prod(facs)
def _default_ctr_fxn(cf,x,ctr_ls):
    facs = [_prodZ(cf,key) for key in ctr_ls] 
    return facs[0]+sum([xi*fi for xi,fi in zip(x,facs[1:])]),facs[1:]
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
    def __init__(self,nsite,hop_ls,decimated,fxn='default',order=1):
        self.nsite = nsite
        self.hop_ls = hop_ls # orbital pair
        self.nparam = len(hop_ls) 
        if decimated is None:
            self.decimated = None
        else:
            self.decimated = list(decimated)
            self.decimated.sort()
        if fxn=='default':
            self.ctr_ls = get_ctr_ls(decimated,order=order)
            def fxn(cf,x):
                return _default_ctr_fxn(cf,x,self.ctr_ls)
            self.nparam += len(self.ctr_ls)-1
        elif fxn is None:
            def fxn(cf,x):
                return 1,None
        else:
            raise ValueError
        self.fxn = fxn

        self.Y = dict()
        for (p,q) in self.hop_ls:
            Y = np.zeros((nsite,)*2)
            Y[p,q] = 1
            self.Y[p,q] = Y-Y.T
    def _update(self,x):
        self.x = np.array(x) # param
        Y = np.zeros((self.nsite,)*2)
        for xi,(p,q) in zip(x,self.hop_ls):
            Y[p,q] = xi
            Y[q,p] = -xi
        self.Y['full'] = Y
    def get_MB_matrix(self,basis):
        n = len(self.hop_ls)
        def fxn(cf):
            return self.fxn(cf,self.x[n:])
        return get_MB_matrix(basis,self.hop_ls,self.x[:n],fxn)
class PMG_autodiff(PMG):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for (p,q),Y in self.Y.items():
            self.Y[p,q] = torch.tensor(Y,requires_grad=False)
    def _update(self,x):
        super()._update(x)
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
class RMG_autodiff(PMG_autodiff):
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
class PMG_manual(PMG):
    def __init__(self,*args,jac_by='frechet',**kwargs):
        super().__init__(*args,**kwargs)
        self.jac_by = jac_by
        nhop = len(self.hop_ls)
        if jac_by=='ad':
            for (p,q),Y in self.Y.items():
                self.Y[p,q] = torch.tensor(Y,requires_grad=False)
            def fxn(x,f):
                Y = sum([xi*self.Y[p,q] for xi,(p,q) in zip(x,self.hop_ls)])
                U = torch.linalg.matrix_exp(f*Y)
                return U,U 
            jac_fxn = torch.func.jacfwd(fxn,has_aux=True)
            def _get_mo_derivative(f):
                x = torch.tensor(self.x[:nhop],requires_grad=True)
                jac,U = jac_fxn(x,f)
                U = U.numpy(force=True)
                jac = jac.numpy(force=True)
                x.grad = None
                return U,jac
        elif jac_by=='frechet':
            def _get_mo_derivative(f):
                Y = self.Y['full']
                jac = np.zeros((self.nsite,self.nsite,nhop),dtype=Y.dtype)
                for i,(p,q) in enumerate(self.hop_ls):
                    U,jac[:,:,i] = scipy.linalg.expm_frechet(Y*f,self.Y[p,q])
                jac *= f
                return U,jac
        else:
            raise ValueError
        self._get_mo_derivative = _get_mo_derivative
    def _update(self,x):
        super()._update(x)
        t0 = time.time()
        U,jac = self._get_mo_derivative(1)
        self.amps = {'f=1':U}
        self.ders = {'f=1':jac}
        if RANK==0:
            print(f'update time=',time.time()-t0)
    def get_mo_derivative(self,cf):
        key = None if self.decimated is None else \
              tuple([cf[p] for p in self.decimated])
        if key in self.ders:
            return self.amps[key],self.ders[key]
        n = len(self.hop_ls)
        f,fder = self.fxn(cf,self.x[n:])
        if f==1:
            return self.amps['f=1'],(self.ders['f=1'],None)
        if f==-1: 
            return self.amps['f=1'].T,(self.ders['f=1'].T,None)
        self.amps[key],jac = self._get_mo_derivative(f)
        self.ders[key] = jac,fder
        return self.amps[key],self.ders[key]
    def get_mo(self,cf,derivative=False):
        if derivative:
            return self.get_mo_derivative(cf)
        key = None if self.decimated is None else \
              tuple([cf[p] for p in self.decimated])
        if key in self.amps:
            return self.amps[key],None
        n = len(self.hop_ls)
        f,_ = self.fxn(cf,self.x[n:])
        if f==1:
            return self.amps['f=1'],None
        if f==-1:
            return self.amps['f=1'].T,None
        self.amps[key] = scipy.linalg.expm(f*self.Y['full'])
        return self.amps[key],None
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
    def add_pmg(self,hop_ls,decimated,remove_redundant=False,**pmg_kwargs):
        hop_ls = self._process_hop_ls(hop_ls,remove_redundant=remove_redundant)
        pmg = self.pmg_class(self.nsite,hop_ls,decimated,**pmg_kwargs)
        self.pmg_ls.append(pmg)
        if RANK==0:
            print('layer=',len(self.pmg_ls)-1)
            print('nparam=',pmg.nparam)
            print('hop_ls=',pmg.hop_ls)
            print()
    def add_mg(self,hop_ls,remove_redundant=False,remove_unphysical=None,**pmg_kwargs):
        hop_ls = self._process_hop_ls(hop_ls,remove_redundant=remove_redundant,remove_unphysical=remove_unphysical)
        pmg_kwargs['fxn'] = None
        mg = self.pmg_class(self.nsite,hop_ls,None,**pmg_kwargs)
        self.pmg_ls.append(mg)
        if RANK==0:
            print('layer=',len(self.pmg_ls)-1)
            print('nparam=',mg.nparam)
            print('hop_ls=',mg.hop_ls)
            print()
    def get_nparam(self):
        self.nparam = sum([pmg.nparam for pmg in self.pmg_ls])
    def _update(self,x):
        for pmg in self.pmg_ls:
            xi,x = x[:pmg.nparam],x[pmg.nparam:]
            pmg._update(xi)
    def get_x(self):
        x = [pmg.x for pmg in self.pmg_ls]
        return np.concatenate(x)
    def _amplitude(self,cf):
        return self._amplitude_and_derivative(cf,derivative=False)[0]
    def _rdm1(self,x):
        n = self.nsite
        cf = [0] * n
        for i,xi in zip(self.decimated,x):
            cf[i] = xi
        _,U = self.get_mo_coeff(cf)
        return U,np.dot(U,U.T.conj())
    def add_rdm1_simple(self,x):
        _,G = self._rdm1(x)
        C = 2*G-np.eye(self.nsite) 
        E = np.array(x)*2-np.ones(len(x))
        C,Z = partial_trace(C,np.diag(E),self.decimated,active=self.active)
        #print(x,Z)
        #print(G)
        #print((C+np.eye(C.shape[0]))/2)
        return (C+np.eye(C.shape[0]))/2,Z
    def rdm1_simple(self):
        nd = len(self.decimated)
        if self.active is None:
            self.active = list(set(range(self.nsite))-set(self.decimated))
            self.active.sort()
        G = 0
        for x in itertools.product((0,1),repeat=nd):
            Gx,Zx = self.add_rdm1_simple(x)
            G += Gx*Zx
        return G
    def add_rdm1(self,x,y):
        if (x,y) in self.rdm_map:
            return self.rdm_map[x,y]
        n = self.nsite
        # compute normalized rdm1:
        # Tr[\hat{\rho}_{yx}'c_p^\dagger c_q]
        Ux,Gx = self._rdm1(x)
        gx = rdm12covariance(Gx)
        #print(x,y)
        err = np.linalg.norm(gx+gx.T)
        if err>1e-10:
            print('skew-symmetry err gx=',err)

        Uy,Gy = self._rdm1(y)
        gy = rdm12covariance(Gy)
        err = np.linalg.norm(gy+gy.T)
        if err>1e-10:
            print('skew-symmetry err gy=',err)

        gyx,tr = covariance_product(gy,gx) 
        err = np.linalg.norm(gyx+gyx.T)
        if err>1e-10:
            print('skew-symmetry err gxy=',err)
        ovlp = np.linalg.det(np.dot(Ux.T,Uy))
        Gyx = covariance2rdm1(gyx)
        #if x==y:
        #    print('ovlp=',ovlp)
        #    g = gyx
        #    print(g[::2,::2])
        #    print(g[1::2,1::2])
        #    print(-1j*g[::2,1::2])
        #    print(1j*g[1::2,::2])
        #    print(x,ovlp)
        #    print(Gyx)
        err = tr-ovlp**2
        if err>1e-10:
            print('tr[xy] err=',err)
        #exit()
        
        Px = np.eye(n)/2
        for i,xi in zip(self.decimated,x):
            Px[i,i] = xi
        gPx = rdm12covariance(Px)
        err = np.linalg.norm(gPx+gPx.T)
        if err>1e-10:
            print('skew-symmetry err gPx=',err)

        gyx_,tr = covariance_product(gyx,gPx) 
        #tr *= 2**4
        err = np.linalg.norm(gyx_+gyx_.T)
        if err>1e-10:
            print('skew-symmetry gxy_=',err)
        Gyx_ = covariance2rdm1(gyx_)

        C = 2*Gyx-np.eye(self.nsite) 
        E = np.array(x)*2-np.ones(len(x))
        _,tr = partial_trace(C,np.diag(E),self.decimated)
        #if x==y:
        #    print(x,tr*ovlp*(2**4))
        #    print(Gyx_)
        #    print((C+np.eye(C.shape[0]))/2)
        #    print(tr)
        self.rdm_map[x,y] = Gyx_,tr*ovlp
        return self.rdm_map[x,y]
    def rdm1(self):
        nd = len(self.decimated)
        self.rdm_map = dict()
        dm = None
        decimated_map = {p:i for i,p in enumerate(self.decimated)}
        for x in itertools.product((0,1),repeat=nd):
            for p,q in itertools.product(range(self.nsite),repeat=2):
                ops = []
                if p in decimated_map:
                    ops.append((decimated_map[p],'des'))
                if q in decimated_map:
                    ops.append((decimated_map[q],'cre'))
                if len(ops)==0:
                    y,sign = tuple(x),1
                else:
                    y,sign = string_act(x,ops,order=1) 
                    if y is None:
                        continue
                    y = tuple(y)
                T,S = self.add_rdm1(x,y)
                if dm is None:
                    dm = T*S
                else:
                    dm[p,q] += T[p,q]*S
        return dm 
class PMGState_autodiff(PMGState):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.U0 is None:
            self.U0_torch = None
        else:
            self.U0 = self.U0[:,:sum(self.nelec)] 
            self.U0_torch = torch.tensor(self.U0,requires_grad=False)
        self.pmg_class = PMG_autodiff
    def get_mo_coeff(self,cf,derivative=False):
        n = len(self.pmg_ls)
        x = [None] * n 
        if derivative:
            dot = torch.matmul 
            U0 = self.U0_torch
        else:
            dot = np.dot
            U0 = self.U0
        for i,pmg in enumerate(self.pmg_ls):
            x[i],Ui = pmg.get_mo(cf,derivative=derivative)
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
        vx = np.concatenate([xi.grad.numpy(force=True) for xi in x])
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
        jacs = [None] * n  
        fders = [None] * n 
        for i,pmg in enumerate(self.pmg_ls):
            Us[i],deri = pmg.get_mo(cf,derivative=derivative)
            if i==0 and lix is not None:
                Us[i] = Us[i][lix]
            if not derivative:
                continue
            jacs[i],fders[i] = deri 
        # left sweep:
        lenv = [Us[0]] + [None] * n 
        for i in range(1,n+1):
            Ui = Us[i]
            if Ui is None:
                assert i==n
                lenv[i] = lenv[i-1][:,:sum(self.nelec)]
            else:
                lenv[i] = np.dot(lenv[i-1],Ui)
        return Us,lenv,jacs,fders
    def _amplitude_and_derivative(self,cf,derivative=True):
        lix = np.argwhere(cf).flatten()
        Us,lenv,jacs,fders = self.get_mo_coeff(cf,derivative=derivative,lix=lix) 
        
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
            jaci,right = jacs[i],renv[i+1]
            if right is not None:
                jaci = np.einsum('jkq,kl->jlq',jaci,right)
            else:
                jaci = jaci[:,:nelec]
            if i>0:
                jaci = np.einsum('ij,jkq->ikq',lenv[i-1],jaci)
            else:
                jaci = jaci[lix]
            jacs[i] = np.einsum('ij,ijq->q',dU,jaci)
        # fders
        for i,pmg in enumerate(self.pmg_ls):
            if fders[i] is None:
                continue
            fderi = np.dot(lenv[i],pmg.Y['full'])
            right = renv[i+1]
            if right is not None:
                fderi = np.dot(fderi,right)
            fderi = np.sum(fderi*dU)
            fders[i] = np.array(fders[i])*fderi
        vx = []
        for jac,fder in zip(jacs,fders):
            vx.append(jac)
            if fder is not None:
                vx.append(fder)
        vx = np.concatenate(vx)
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
