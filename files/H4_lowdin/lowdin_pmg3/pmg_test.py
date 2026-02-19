import numpy as np
import scipy.linalg
import pickle,h5py,itertools
from PyPMG.h4_model import * 
from PyPMG.vmc import *
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(precision=10,suppress=True)

penalty = False 
HF_typ = 'GHF'
symmetry = 'u1'
pmg_typ = 3 
remove_redundant = True

U0 = np.zeros((8,8))
U0[::2,::2] = U0[1::2,1::2] = np.array([[1, 1, 1, 1.],
                                        [1, 1,-1,-1],
                                        [1,-1, 1,-1],
                                        [1,-1,-1, 1]])/2.
eps = 0.1
#eps = 1
psi = H4MinimalState(HF_typ,pmg_typ=pmg_typ,U0=U0,remove_redundant=remove_redundant,symmetry=symmetry)

#x = (np.random.rand(psi.nparam)*2-1)*eps
#np.save('x.npy',x)
#exit()
x = np.load('x.npy')
psi._update(x)
G = psi.rdm1_simple()
G = psi.rdm1()
print(G)
G_ = np.zeros_like(G)
for p,q in itertools.product(range(psi.nsite),repeat=2):
    for x in itertools.product((0,1),repeat=psi.nsite):
        psi_x = psi.amplitude(x)
        if np.absolute(psi_x)<1e-10:
            continue
        #ops = (p,'cre'),(q,'des')
        ops = (p,'des'),(q,'cre')
        y,sign = string_act(x,ops,order=1)
        if y is None:
            continue
        psi_y = psi.amplitude(y)
        G_[p,q] += psi_x*psi_y*sign
print(G_)
