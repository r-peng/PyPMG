import numpy as np
import scipy,itertools
np.set_printoptions(precision=6,suppress=True)
def occ2rdm(occ):
    return np.diag(np.array(occ)) 
def mo2rdm(mo):
    return np.dot(mo,mo.T.conj())
def rdm2corr(rdm):
    return rdm*2-np.eye(rdm.shape[0])
def occ2corr(occ):
    rdm = occ2rdm(occ)
    return rdm2corr(rdm)
def mo2corr(mo):
    rdm = mo2rdm(mo)
    return rdm2corr(rdm)
class MGState:
    def __init__(self,mo,nin):
        # modes order: c2,c4,c1,c3
        #     d1,d2
        # c2 [  ,  ]
        # c4 [  ,  ]
        # c1 [  ,  ]
        # c3 [  ,  ]
        C = mo2corr(mo)
        self.D = C[:nin,:nin]
        self.Bdag = C[:nin,nin:]
        self.B = C[nin:,:nin]
        self.A = C[nin:,nin:]
    def Cin2Cout(self,Cin):
        Cout = np.linalg.inv(self.D+Cin)
        Cout = np.dot(self.B,np.dot(Cout,self.Bdag))
        return self.A-Cout
    def sample(self,occ):
        Cin = occ2corr(occ)
        print(Cin)
        Cout = self.Cin2Cout(Cin)
        print(Cout)
        w,v = np.linalg.eigh(Cout)
        print(w)
class H2State:
    def __init__(self,h,mo):
        self.mgs = MGState(mo,2)
        self.h = h 
    def sample(self,n):
        self.mgs.sample(n[:2])
if __name__=='__main__':
    h = None
    K = np.random.rand(4,4)*2-1
    K = K + 1j*(np.random.rand(4,4)*2-1)
    K -= K.T.conj()
    mo = scipy.linalg.expm(K)
    h2 = H2State(h,mo[:,:2])
    for i,j in itertools.product(range(2),repeat=2):
        print(i,j)
        h2.sample([i,j,0,0])
        print()
