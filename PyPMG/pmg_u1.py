import numpy as np
import scipy,itertools
import torch
np.set_printoptions(precision=6,suppress=True)
def _fermion_act(x,oix,typ):
    kill = {'cre':1,'des':0}[typ]
    if x[oix]==kill:
        return None,0
    sign = (-1)**sum(x[:oix])
    y = x.copy()
    y[oix] = 1-x[oix] 
    return y,sign
def _string_act(x,ops,order=-1):
    if order==-1:
        ops = ops[::-1]
    s = np.zeros(len(ops),dtype=int)
    y = x.copy()
    for i,(oix,typ) in enumerate(ops):
        y,s[i] = _fermion_act(y,oix,typ)
        if y is None:
            return None,0
    return tuple(y),np.prod(s)
class H2State:
    def __init__(self,h,kvec):
        # modes order: c1,c2,c3,c4
        self.nsite = 4
        self.nparam = 1+len(kvec)
        self.idxs = [(i,j) for i in range(self.nsite) for j in range(i+1,self.nsite)]
        assert len(self.idxs)==len(kvec)
        self._update(h,kvec)
    def get_all_configs(self):
        configs = []
        for cf in itertools.product((0,1),repeat=self.nsite):
            if sum(cf)!=2:
                continue
            if sum(cf[::2])!=1:
                continue
            if sum(cf[1::2])!=1:
                continue
            configs.append(cf)
        return configs
    def new_configs(self,x):
        y1 = x.copy()
        y1[::2] = 1-x[::2]
        y2 = x.copy()
        y2[1::2] = 1-x[1::2]
        return y1,y2 
    def get_x(self,h=None,kvec=None):
        if h is None:
            h = self.h
        if kvec is None:
            kvec = self.kvec
        return np.concatenate([np.ones(1)*h,kvec])
    def update(self,x):
        self._update(x[0],x[1:])
    def _update(self,h,kvec):
        self.hmat = np.zeros((4,4))
        self.hmat[0,2] = 1
        self.hmat[2,0] = -1
        self.h = h
        self.left = scipy.linalg.expm(-h*self.hmat)

        self.kvec = kvec
        self.kappa = np.zeros((4,4))
        for ix,(i,j) in enumerate(self.idxs):
            self.kappa[i,j] = kvec[ix]
        self.right = scipy.linalg.expm(self.kappa-self.kappa.T)
    def amplitude_and_derivative(self,x):
        if sum(x)!=2:
            return 0,None

        h = torch.tensor(self.h,requires_grad=True)
        #kvec = torch.tensor(self.kvec,requires_grad=True)
        kappa = torch.tensor(self.kappa,requires_grad=True)
        hmat = torch.tensor(self.hmat,requires_grad=False)

        h_ = h*(-1)**x[1]
        left = torch.linalg.matrix_exp(-h_*hmat)
        idx = np.argwhere(x).flatten()
        left = left[:,idx]

        right = torch.linalg.matrix_exp(kappa-kappa.T)

        psi_x = torch.matmul(left.T,right[:,:2])
        psi_x = torch.linalg.det(psi_x)

        psi_x.backward()
        vx = kappa.grad.numpy(force=True)
        vx = np.array([vx[i,j] for i,j in self.idxs])
        vx = self.get_x(h=h.grad.numpy(force=True),kvec=vx)
        psi_x = psi_x.numpy(force=True)
        return psi_x,vx
    def amplitude(self,x):
        if sum(x)!=2:
            return 0
        left = self.left if x[1]==0 else self.left.T
        idx = np.argwhere(x).flatten()
        left = left[:,idx]

        M = np.dot(left.T,self.right[:,:2])
        return np.linalg.det(M)
    def log_prob(self,x):
        if sum(x)!=2:
            return None
        psi_x = self.amplitude(x)
        return np.log(psi_x**2)
class Ham:
    def __init__(self,hcore,eri):
        self.hcore = hcore
        self.eri = eri
        self.nmo = hcore.shape[0]
        self.thresh = 1e-6
    def eloc_terms(self,x):
        terms = {}
        x = np.array(x)
        for (p,q) in itertools.product(range(self.nmo),repeat=2):
            ops = (p,'cre'),(q,'des')
            y,sign = _string_act(x,ops)
            if y is None:
                continue
            coeff = sign*self.hcore[p,q]
            if np.fabs(coeff)>self.thresh:
                if y not in terms:
                    terms[y] = 0
                terms[y] += coeff
        for (p,q,r,s) in itertools.product(range(self.nmo),repeat=4):
            ops = (p,'cre'),(q,'cre'),(s,'des'),(r,'des')
            y,sign = _string_act(x,ops)
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
class VMC:
    def __init__(self,psi,ham):
        self.psi = psi
        self.ham = ham
        self.thresh = 1e-10
    def eloc(self,x,derivative=True):
        if derivative:
            psi_x,vx = self.psi.amplitude_and_derivative(x)
        else:
            psi_x = self.psi.amplitude(x)
            vx = 0
        if np.absolute(psi_x)<self.thresh:
            return psi_x,0,None
        terms = self.ham.eloc_terms(x)
        eloc = 0
        for (y,coeff) in terms.items():
            eloc += self.psi.amplitude(y)*coeff
        return psi_x,eloc/psi_x,vx/psi_x
    def sample_exact(self):
        E = []
        v = []
        p = []
        for x in itertools.product((0,1),repeat=self.psi.nsite):
            psi_x,ex,vx = self.eloc(x) 
            px = psi_x**2
            if px<self.thresh:
                continue
            E.append(ex)
            v.append(vx)
            p.append(px)
        self.p = np.array(p)
        self.E = np.array(E)
        self.v = np.array(v)

        nsq = self.p.sum()
        self.p /= nsq
        self.energy = np.dot(self.E,self.p)
    def SR(self,cond=1e-3):
        vmean = np.dot(self.p,self.v)
        g = np.dot(self.E*self.p,self.v)-self.energy*vmean 
        S = np.einsum('i,ij,ik->jk',self.p,self.v,self.v)
        S -= np.outer(vmean,vmean)
        S += np.eye(len(vmean))*cond
        return np.dot(np.linalg.inv(S),g)
    def run(self,nsteps,stepsize,thresh=1e-6):
        e_old = 0
        for i in range(nsteps):
            self.sample_exact()
            print(f'step={i},energy={self.energy}')
            if e_old-self.energy<0:
                raise ValueError
            if e_old-self.energy<thresh:
                return
            e_old = self.energy
            dx = self.SR()
            xnew = self.psi.get_x()-stepsize*dx
            self.psi.update(xnew)
if __name__=='__main__':
    N = 4
    h = np.random.rand()
    kvec = np.random.rand(N*(N-1)//2)
    psi = H2State(h,kvec)
    for i,j,k,l in itertools.product((0,1),repeat=4):
        x = i,j,k,l
        amp1 = psi.amplitude(x)
        amp2,_ = psi.amplitude_and_derivative(x)
        err = np.fabs(amp1-amp2)
        if err>1e-10:
            print(x,amp1,amp2)

