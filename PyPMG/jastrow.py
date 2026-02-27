import numpy as np
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
