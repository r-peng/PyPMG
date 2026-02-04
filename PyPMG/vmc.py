import time,scipy,functools,h5py,gc
import numpy as np
import scipy.sparse.linalg as spla
import scipy.optimize as opt
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
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
        self.Lold = None
        for step in range(start,stop):
            self.step = step
            if RANK==0:
                print('step=',step)
                print('\nnparam=',self.nparam)
            self.sample()
            self.extract_energy_gradient()
            if RANK==0:
                dE = 0 if self.Eold is None else self.E-self.Eold
                print(f'energy={self.E},dE={dE},std={self.Eerr}')
                self.Eold = self.E

                dL = 0 if self.Lold is None else self.L-self.Lold
                print(f'loss  ={self.L},dloss={dL},std={self.Lerr}')
                self.Lold = self.L
                print(f'\tgnorm=',np.linalg.norm(self.g))
            x = self.transform_gradients()
            self.free_quantities()
            COMM.Bcast(x,root=0) 
            self.psi.update(x)
            if save_wfn and RANK==0:
                np.save(f'psi{step+1}.npy',x)
    def sample(self,sample_size=None,compute_v=True,compute_h=None,save_config=True):
        ham = self.ham['energy']
        self.sampler.preprocess(self.psi,ham=ham)
        compute_h = self.compute_h if compute_h is None else compute_h
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
        while recv<sample_size:
            rank,step = COMM.recv(tag=0)
            if step!=self.step:
                raise ValueError(f'step={step},self.step={self.step}')
            recv += 1
            send += 1
            COMM.send(send,dest=rank)
        print('\tsample time=',time.time()-t0)
    def _accumulate(self,x,compute_v,compute_h):
        self.cfs.append(x)
        for key in self.ham:
            self.ham[key].compute_eloc(x,self.psi)
        if compute_v:
            self.psi.amplitude_and_derivative(x)
    def _collect(self,compute_v,compute_h):
        self.c = np.array([self.psi.amps[x] for x in self.cfs])
        self.psi.amps = dict()

        self.e = {'loss':0} 
        for key in self.ham:
            self.e[key] = np.array([self.ham[key].elocs[x] for x in self.cfs])
            self.e['loss'] += self.ham[key].weight*self.e[key]
            self.ham[key].elocs = dict()
        if compute_v:
            self.v = np.array([self.psi.ders[x] for x in self.cfs])
        self.psi.ders = dict()
    def _sample_stochastic(self,sample_size,compute_v,compute_h):
        self.cfs = []
        ham = self.ham['energy']
        while True:
            send = COMM.recv(source=0)
            if send>=sample_size:
                break 
            x,_ = self.sampler.sample(self.psi,ham=ham)
            cx = self.psi.amplitude(x)
            if cx is None or np.fabs(cx) < self.psi.thresh:
                continue
            self._accumulate(x,compute_v,compute_h)
            COMM.send((RANK,self.step),dest=0,tag=0) 
        self._collect(compute_v,compute_h)
    def _sample_exact(self,compute_v,compute_h): 
        p = self.sampler.p
        all_cfs = self.sampler.all_cfs
        ixs = self.sampler.nonzeros
        #print(all_cfs,ixs)
        ntotal = len(ixs)
        if RANK==SIZE-1:
            print('\tnsamples per process=',ntotal)

        self.cfs = []
        for ix in ixs:
            x = all_cfs[ix]
            cx = self.psi.amplitude(x)
            if cx is None:
                raise ValueError
            self._accumulate(x,compute_v,compute_h)
        self._collect(compute_v,compute_h)
    def extract_energy_gradient(self):
        t0 = time.time()
        self.extract_energy()
        self.extract_gradient()
        if self.optimizer in ['rgn','lin','trust']:
            self._extract_hmean()
        if RANK==0:
            print('\tcollect g,h time=',time.time()-t0)
    def extract_energy(self):
        if RANK>0:
            if self.sampler.exact:
                self.f = self.c*self.c.conj() 
            else:
                self.f = np.ones(len(self.c))
            COMM.send((self.e['loss'],self.e['energy'],self.f),dest=0,tag=1)
            return
        l = []
        e = []
        f = []
        for worker in range(1,SIZE):
            li,ei,fi = COMM.recv(source=worker,tag=1)
            l.append(li)
            e.append(ei)
            f.append(fi)
        l = np.concatenate(l)
        e = np.concatenate(e)
        self.f = np.concatenate(f)
        self.n = self.f.sum()
        print('all_e=',e)
        if self.sampler.exact:
            print('all_f=',self.f)
            self.L = np.dot(self.f,l)
            self.E = np.dot(self.f,e)
            self.Lerr,self.Eerr = 0,0
        else:
            print('nsamples=',self.n)
            self.E,self.Eerr = blocking_analysis(e)
            self.L,self.Lerr = blocking_analysis(l)
    def extract_gradient(self):
        self.vmean = np.zeros(self.nparam,dtype=self.dtype)
        vsum = self.vmean.copy() if RANK==0 else\
               np.dot(self.f,self.v) 
        COMM.Reduce(vsum,self.vmean,op=MPI.SUM,root=0)

        self.evmean = np.zeros(self.nparam,dtype=self.dtype)
        evsum = self.evmean.copy() if RANK==0 else\
                np.dot(self.f*self.e['loss'],self.v) 
        COMM.Reduce(evsum,self.evmean,op=MPI.SUM,root=0)
        if RANK>0:
            self.vmean = None
            self.evmean = None
            #COMM.send(self.v,dest=0,tag=10)
            #exit()
            return 
        self.vmean /= self.n
        self.evmean /= self.n
        self.g = (self.evmean - self.L.conj() * self.vmean).real
        #v = COMM.recv(source=1,tag=10)
        #v -= self.vmean.reshape(1,self.nparam)
        #v *= self.f.reshape(len(self.f),1)
        ##q,r = np.linalg.qr(v)
        #q,r,p = scipy.linalg.qr(v,pivoting=True)
        #print('p=',p)
        #for i in range(r.shape[1]):
        #    ri = r[:,i]
        #    print('i=',i,ri)
        #    ri = np.fabs(ri)
        #    li = len(ri[ri>1e-6])
        #    print(li,p[:li])
        #exit()
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
        vvmean = np.zeros((self.nparam,)*2,dtype=self.dtype)
        vvsum = vvmean.copy() if RANK==0 else\
                np.einsum('s,si,sj->ij',self.f,self.v.conj(),self.v)
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
                Sx1 = np.dot(self.f*np.dot(v,x),v.conj())
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
    def initialize(self,psi):
        self.all_cfs = psi.get_all_configs()
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
    def preprocess(self,psi,ham=None):
        if self.all_cfs is None:
            self.initialize(psi)
        self.compute_dense_prob(psi)
    def compute_dense_prob(self,psi):
        t0 = time.time()
        ptot = np.zeros(self.ntot)
        start,stop = self.start,self.stop
        cfs = self.all_cfs[start:stop]

        plocal = [] 
        for x in cfs:
            px = psi.log_prob(x)
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
        print(self.p)

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
    def __init__(self,burn_in=40,every=10,seed=None):
        self.burn_in = burn_in
        self.rng = np.random.default_rng(seed)
        self.every = every
        self.exact = False
    def preprocess(self,psi,ham=None):
        self._burn_in(psi,ham=ham)
    def _burn_in(self,psi,ham=None,cf=None,burn_in=None,exclude_root=True):
        if cf is not None:
            self.cf = cf 
        self.px = psi.log_prob(self.cf)

        if exclude_root and RANK==0:
            print('\tlog prob=',self.px)
            #exit()
            return 
        burn_in = self.burn_in if burn_in is None else burn_in
        t0 = time.time()
        for n in range(burn_in):
            self.cf,_ = self.sample(psi,ham=ham)
        if RANK==SIZE-1:
            print('\tburn in time=',time.time()-t0)
    def sample(self,psi,ham=None):
        for _ in range(self.every):
            y,qx2y = psi.propose(self.cf,self.rng,ham=ham)
            qy2x = psi.propose_reverse(self.cf,y,ham=ham)
            py = psi.log_prob(y)
            if py is None:
                continue
            acceptance = np.exp(py-self.px)
            acceptance *= qy2x/qx2y
            if acceptance<self.rng.uniform():
                continue
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
