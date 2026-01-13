import time,scipy,functools,h5py,gc
import numpy as np
import scipy.sparse.linalg as spla
import scipy.optimize as opt
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
def compute_local_energy(x,psi,ham,compute_v=True,compute_h=False):
    if compute_v:
        psi_x,vx = psi.amplitude_and_derivative(x)
    else:
        psi_x = psi.amplitude(x)
        vx = 0
    terms = ham.eloc_terms(x)
    eloc = 0
    for (y,coeff) in terms.items():
        eloc += psi.amplitude(y)*coeff
    return psi_x,eloc/psi_x,vx/psi_x,None
def blocking_analysis(energies, weights=None, neql=0, printQ=True):
    weights = np.ones_like(energies) if weights is None else weights
    nSamples = weights.shape[0] - neql
    weights = weights[neql:]
    energies = energies[neql:]
    weightedEnergies = np.multiply(weights, energies)
    meanEnergy = weightedEnergies.sum() / weights.sum()
    if printQ:
        print(f'\nMean energy: {meanEnergy:.8e}')
        print('Block size    # of blocks        Mean                Error')
    blockSizes = np.array([ 1, 2, 5, 10, 20, 50, 70, 100, 200, 500 ])
    prevError = 0.
    plateauError = None
    for i in blockSizes[blockSizes < nSamples/2.]:
        nBlocks = nSamples//i
        blockedWeights = np.zeros(nBlocks)
        blockedEnergies = np.zeros(nBlocks)
        for j in range(nBlocks):
            blockedWeights[j] = weights[j*i:(j+1)*i].sum()
            blockedEnergies[j] = weightedEnergies[j*i:(j+1)*i].sum() / blockedWeights[j]
        v1 = blockedWeights.sum()
        v2 = (blockedWeights**2).sum()
        mean = np.multiply(blockedWeights, blockedEnergies).sum() / v1
        error = (np.multiply(blockedWeights, (blockedEnergies - mean)**2).sum() / (v1 - v2 / v1) / (nBlocks - 1))**0.5
        if printQ:
            print(f'  {i:4d}           {nBlocks:4d}       {mean:.8e}       {error:.6e}')
        if error < 1.05 * prevError and plateauError is None:
            plateauError = max(error, prevError)
        prevError = error

    #print(RANK,plateauError,error)
    if plateauError is None:
        plateauError = error
    else:
        if printQ:
            print(f'Stocahstic error estimate: {plateauError:.6e}\n')

    return meanEnergy, plateauError
class SGD: # stochastic sampling
    def __init__(self,psi,ham,sampler,optimizer='sgd',solve_dense=True):
        # parse sampler
        self.psi = psi
        self.ham = ham
        self.sampler = sampler

        # parse wfn 
        self.nsite = psi.nsite
        x = psi.get_x()
        self.dtype = x.dtype
        self.nparam = x.size 

        # parse gradient optimizer
        self.optimizer = optimizer
        self.solve_dense = solve_dense
        self.compute_h = False
        self.maxiter = 500 
        self.discard = 1e3
        self.tol = 1e-4

        # to be set before run
        self.tmpdir = None
        self.sample_size = None
        self.rate1 = None # rate for SGD,SR
        self.ctr_update = None

        self.free_g = True
    def free_quantities(self):
        if RANK==0: 
            self.Eold = self.E
        self.f = None
        self.e = None
        if self.free_g:
            self.g = None
        self.v = None
        self.vmean = None
        self.evmean = None
        self.vsum = None
        self.evsum = None
        self.h = None
        self.hmean = None
        self.hsum = None

        self.S = None
        self.vvmean = None
        self.H = None
        self.vhmean = None
        self.Sx1 = None
        self.Hx1 = None
        self.Hx1h = None
        gc.collect()
    def run(self,start,stop,save_wfn=True):
        self.Eold = None 
        for step in range(start,stop):
            self.step = step
            if RANK==0:
                print('\nnparam=',self.nparam)
            self.sample()
            self.extract_energy_gradient()
            x = self.transform_gradients()
            self.free_quantities()
            COMM.Bcast(x,root=0) 
            self.psi.update(x)
    def sample(self,sample_size=None,compute_v=True,compute_h=None,save_config=True):
        self.sampler.preprocess(self.psi)
        compute_h = self.compute_h if compute_h is None else compute_h
        self.c = []
        self.e = []
        if compute_v:
            self.evsum = np.zeros(self.nparam,dtype=self.dtype)
            self.vsum = np.zeros(self.nparam,dtype=self.dtype)
            self.v = []
        if compute_h:
            self.hsum = np.zeros(self.nparam,dtype=self.dtype)
            self.h = [] 
        if self.sampler.exact:
            if RANK==0:
                return
            self._sample_exact(compute_v,compute_h)
            return

        sample_size = self.sample_size if sample_size is None else sample_size
        if RANK==0:
            configs = self._ctr(sample_size)
            #if save_config:
            #    np.save(self.tmpdir+f'config{self.step}.npy',np.array(configs[-SIZE:]))
        else:
            self._sample_stochastic(sample_size,compute_v,compute_h)
    def _ctr(self,sample_size):
        t0 = time.time()
        recv = 0
        for send in range(SIZE-1):
            COMM.send(send,dest=send+1)
        configs = []
        while recv<sample_size:
            rank,step,config = COMM.recv(tag=0)
            if step!=self.step:
                raise ValueError(f'step={step},self.step={self.step}')
            recv += 1
            configs.append(config)
            send += 1
            COMM.send(send,dest=rank)
        print('\tsample time=',time.time()-t0)
        #print(configs)
        return configs
    def _sample_stochastic(self,sample_size,compute_v,compute_h):
        self.f = None
        while True:
            send = COMM.recv(source=0)
            if send>=sample_size:
                break 
            cf,_ = self.sampler.sample()
            cx,ex,vx,hx = compute_local_energy(cf,self.psi,self.ham,compute_v=compute_v,compute_h=compute_h)
            if cx is None or np.fabs(ex.real) > self.discard:
                print(f'RANK={RANK},cx={cx},ex={ex}')
                ex = np.zeros(1)[0]
                err = 0.
                if compute_v:
                    vx = np.zeros(self.nparam,dtype=self.dtype)
                if compute_h:
                    hx = np.zeros(self.nparam,dtype=self.dtype)
            self.c.append(cx)
            self.e.append(ex)
            if compute_v:
                self.vsum += vx
                self.evsum += vx * ex.conj()
                self.v.append(vx)
            if compute_h:
                self.hsum += hx
                self.h.append(hx)
            COMM.send((RANK,self.step,cf),dest=0,tag=0) 

        #self.sampler.config = self.config
        self.e = np.array(self.e)
        self.c = np.array(self.c)
        if compute_v:
            self.v = np.array(self.v)
        if compute_h:
            self.h = np.array(self.h)
    def _sample_exact(self,compute_v,compute_h): 
        self.f = []
        p = self.sampler.p
        all_cfs = self.sampler.all_cfs
        ixs = self.sampler.nonzeros
        ntotal = len(ixs)
        if RANK==SIZE-1:
            print('\tnsamples per process=',ntotal)
        for ix in ixs:
            cf = all_cfs[ix]
            cx,ex,vx,hx = compute_local_energy(cf,self.psi,self.ham,compute_v=compute_v,compute_h=compute_h)
            if cx is None:
                raise ValueError
            if np.fabs(ex.real)*p[ix] > self.discard:
                raise ValueError(f'RANK={RANK},config={config},cx={cx},ex={ex}')
            self.f.append(p[ix])
            self.e.append(ex)
            if compute_v:
                self.vsum += vx * p[ix]
                self.evsum += vx * ex.conj() * p[ix]
                self.v.append(vx)
            if compute_h:
                self.hsum += hx * p[ix]
                self.h.append(hx)
        self.f = np.array(self.f)
        self.e = np.array(self.e)
        self.c = np.array(self.c)
        if compute_v:
            self.v = np.array(self.v)
        if compute_h:
            self.h = np.array(self.h)
    def extract_energy_gradient(self):
        t0 = time.time()
        self.extract_energy()
        self.extract_gradient()
        if self.optimizer in ['rgn','lin','trust']:
            self._extract_hmean()
        if RANK==0:
            try:
                dE = 0 if self.Eold is None else self.E-self.Eold
                print(f'step={self.step},E={self.E},dE={dE},std={self.Eerr}')
                print(f'\tgnorm=',np.linalg.norm(self.g))
            except TypeError:
                print('E=',self.E)
            print('\tcollect g,h time=',time.time()-t0)
    def extract_energy(self):
        if RANK>0:
            COMM.send((self.e,self.f),dest=0,tag=1)
            return
        e = []
        f = []
        for worker in range(1,SIZE):
            ei,fi = COMM.recv(source=worker,tag=1)
            e.append(ei)
            if fi is not None:
                f.append(fi)
        e = np.concatenate(e)
        #print('all_e=',e)
        if fi is not None:
            f = np.concatenate(f)
            #print('all_f=',f)
            self.E = np.dot(f,e)
            self.Eerr,self.n = 0,1
        else:
            self.n = len(e)
            print('nsamples=',self.n)
            self.E,self.Eerr = blocking_analysis(e)
    def extract_gradient(self):
        vmean = np.zeros(self.nparam,dtype=self.dtype)
        COMM.Reduce(self.vsum,vmean,op=MPI.SUM,root=0)
        evmean = np.zeros(self.nparam,dtype=self.dtype)
        COMM.Reduce(self.evsum,evmean,op=MPI.SUM,root=0)
        if RANK>0:
            return 
        self.vmean = vmean/self.n
        self.evmean = evmean/self.n
        self.g = (self.evmean - self.E.conj() * self.vmean).real
    def update(self,deltas):
        x = self.psi.get_x()
        xnorm = np.linalg.norm(x)
        dnorm = np.linalg.norm(deltas) 
        print(f'\txnorm={xnorm},dnorm={dnorm}')
        if self.ctr_update is not None:
            tg = self.ctr_update * xnorm
            if dnorm > tg:
                deltas *= tg/dnorm
        return x - deltas
    def transform_gradients(self):
        if RANK>0:
            return np.zeros(self.nparam,dtype=self.dtype) 
        if self.optimizer=='sgd':
            deltas = self.g
        elif self.optimizer=='sign':
            deltas = np.sign(self.g)
        elif self.optimizer=='signu':
            deltas = np.sign(self.g) * np.random.uniform(size=self.nparam)
        else:
            raise NotImplementedError
        return self.update(self.rate1*deltas)
class SR(SGD):
    def __init__(self,psi,ham,sampler,**kwargs):
        super().__init__(psi,ham,sampler,**kwargs) 
        self.optimizer = 'sr' 
        self.eigen_thresh = None
        self.cond1 = None
    def _get_Smatrix(self):
        t0 = time.time()
        if RANK==0:
            vvsum = np.zeros((self.nparam,)*2,dtype=self.dtype)
        else:
            v = self.v
            vvsum = np.dot(v.T.conj(),v) if self.f is None else\
                    np.einsum('s,si,sj->ij',self.f,v.conj(),v)
        vvmean = np.zeros_like(vvsum)
        COMM.Reduce(vvsum,vvmean,op=MPI.SUM,root=0)
        if RANK>0:
            return
        vmean = self.vmean
        self.S = vvmean/self.n - np.outer(vmean.conj(),vmean)
        print('\tcollect S matrix time=',time.time()-t0)
    def _get_S_iterative(self):
        self.Sx1 = np.zeros(self.nparam,dtype=self.dtype)
        vmean = self.vmean
        v = self.v
        if RANK==0:
            def matvec(x):
                COMM.Bcast(self.terminate,root=0)
                COMM.Bcast(x,root=0)
                Sx1 = np.zeros_like(self.Sx1)
                COMM.Reduce(Sx1,self.Sx1,op=MPI.SUM,root=0)     
                return self.Sx1/self.n-(vmean.conj()*np.dot(vmean,x))
        else: 
            def matvec(x):
                COMM.Bcast(self.terminate,root=0)
                if self.terminate[0]==1:
                    return 0 
                COMM.Bcast(x,root=0)
                Sx1 = np.dot(np.dot(v,x),v.conj()) if self.f is None else \
                      np.dot(self.f*np.dot(v,x),v.conj())
                COMM.Reduce(Sx1,self.Sx1,op=MPI.SUM,root=0)     
                return 0 
        return matvec
    def transform_gradients(self):
        deltas = self._transform_gradients_sr(self.solve_dense)
        if RANK>0:
            return deltas
        return self.update(self.rate1*deltas)
    def _transform_gradients_sr(self,solve_dense):
        if not solve_dense:
            self._get_S_iterative()
            return self._transform_gradients_sr_iterative()

        self._get_Smatrix()
        if RANK>0:
            return np.zeros(self.nparam,dtype=self.dtype) 

        t0 = time.time()
        if self.eigen_thresh is None:
            deltas = np.linalg.solve(self.S + self.cond1 * np.eye(self.nparam),self.g)
        else:
            w,v = np.linalg.eigh(self.S)
            w = w[w>self.eigen_thresh*w[-1]]
            print(f'\tnonzero={len(w)},wmax={w[-1]}')
            v = v[:,-len(w):]
            deltas = np.dot(v/w.reshape(1,len(w)),np.dot(v.T,self.g)) 
        print('\tSR solver time=',time.time()-t0)
        return deltas
    def _transform_gradients_sr_iterative(self):
        g = self.g if RANK==0 else np.zeros(self.nparam,dtype=self.dtype)
        def R(x):
            return self.S(x) + self.cond1 * x
        deltas = self.solve_iterative(R,g,True,x0=g)
        if RANK>0:
            return np.zeros(self.nparam,dtype=self.dtype)  
        return deltas
    def solve_iterative(self,A,b,symm,x0=None):
        self.terminate = np.array([0])
        deltas = np.zeros_like(b)
        sh = len(b)
        if RANK==0:
            print('symmetric=',symm)
            t0 = time.time()
            LinOp = spla.LinearOperator((sh,sh),matvec=A,dtype=b.dtype)
            if symm:
                deltas,info = spla.minres(LinOp,b,x0=x0,tol=self.tol,maxiter=self.maxiter)
            else: 
                deltas,info = self.solver(LinOp,b,x0=x0,tol=self.tol,maxiter=self.maxiter)
            self.terminate[0] = 1
            COMM.Bcast(self.terminate,root=0)
            print(f'\tsolver time={time.time()-t0},exit status={info}')
        else:
            nit = 0
            while self.terminate[0]==0:
                nit += 1
                A(deltas)
            if RANK==1:
                print('niter=',nit)
        return deltas
##############################################################################################
# sampler
#############################################################################################
import itertools
class DenseSampler:
    def __init__(self,exact=False,seed=None,thresh=1e-28):
        self.p = None
        self.rng = np.random.default_rng(seed)
        self.exact = exact 
        self.thresh = thresh
        self.all_cfs = None
    def initialize(self):
        self.all_cfs = self.psi.get_all_configs()
        self.ntot = len(self.all_cfs)
        if RANK==0:
            print('ntotal configs=',self.ntot)
        self.flat_idx = list(range(self.ntot))

        batchsize,remain = self.ntot//SIZE,self.ntot%SIZE
        self.count = np.array([batchsize]*SIZE)
        if remain > 0:
            self.count[-remain:] += 1
        self.disp = np.concatenate([np.array([0]),np.cumsum(self.count[:-1])])
        self.start = self.disp[RANK]
        self.stop = self.start + self.count[RANK]
    def preprocess(self,psi):
        self.psi = psi
        if self.all_cfs is None:
            self.initialize()
        self.compute_dense_prob()
    def compute_dense_prob(self):
        t0 = time.time()
        ptot = np.zeros(self.ntot)
        start,stop = self.start,self.stop
        cfs = self.all_cfs[start:stop]

        plocal = [] 
        for x in cfs:
            px = self.psi.log_prob(x)
            px = 0 if px is None else np.exp(px) 
            plocal.append(px)
        plocal = np.array(plocal)
         
        COMM.Allgatherv(plocal,[ptot,self.count,self.disp,MPI.DOUBLE])
        nonzeros = []
        for ix,px in enumerate(ptot):
            if px > self.thresh:
                nonzeros.append(ix) 
        n = np.sum(ptot)
        self.p = ptot/n

        ntot = len(nonzeros)
        batchsize,remain = ntot//(SIZE-1),ntot%(SIZE-1)
        L = SIZE-1-remain
        if RANK-1<L:
            start = (RANK-1)*batchsize
            stop = start+batchsize
        else:
            start = (batchsize+1)*(RANK-1)-L
            stop = start+batchsize+1
        self.nonzeros = nonzeros if RANK==0 else nonzeros[start:stop]
        if RANK==SIZE-1:
            print('\tdense amplitude time=',time.time()-t0)
            print('\ttotal non-zero amplitudes=',ntot)
    def sample(self):
        idx = self.rng.choice(self.flat_idx,p=self.p)
        config = self.all_cfs[idx]
        omega = self.p[idx]
        return config,omega
class MHSampler:
    def __init__(self,burn_in=40,seed=None):
        self.burn_in = burn_in
        self.rng = np.random.default_rng(seed)
        self.exact = False
    def preprocess(self,psi):
        self.psi = psi
        self._burn_in()
    def _burn_in(self,cf=None,burn_in=None,exclude_root=True):
        if cf is not None:
            self.cf = cf 
        self.px = self.psi.log_prob(self.cf)

        if exclude_root and RANK==0:
            print('\tlog prob=',self.px)
            #exit()
            return 
        burn_in = self.burn_in if burn_in is None else burn_in
        t0 = time.time()
        for n in range(burn_in):
            self.cf,_ = self.sample()
        if RANK==SIZE-1:
            print('\tburn in time=',time.time()-t0)
    def sample(self):
        cfs = self.psi.new_configs(self.cf)
        y = cfs[self.rng.choice(len(cfs))]
        py = self.psi.log_prob(y)
        if py is None:
            return self.cf,self.px
        acceptance = np.exp(py-self.px)
        if acceptance<self.rng.uniform():
            return self.cf,self.px
        self.cf,self.px = y,py
        return self.cf,self.px
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
