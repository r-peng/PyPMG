import numpy as np
from PyMPG.HF import hartree_fock_spin_orbital 

L = 6
t = 1.0
PBC = True

N_up = 2
N_dn = 2
N_e = N_up + N_dn
nbar = N_e / L

HFLIP = 0.5

# Building Hamiltonian
# index by u1,d1,u2,d2,...
H = np.zeros((L*2,)*2)
for i in range(L):
    j = (i+1)%L
    # hopping
    for s in (0,1):
        ix1,ix2 = 2*i+s,2*j+s
        H[ix1,ix2] = H[ix2,ix1] = -t
    # spin-flip
    ix1,ix2 = 2*i,2*i+1
    H[ix1,ix2] = H[ix2,ix1] = HFLIP 
V = np.zeros((L*2,)*4)
for i in range(L):
    ix1,ix2 = 2*i,2*i+1
    V[ix1,ix1,ix2,ix2] = V[ix2,ix2,ix1,ix1] = U

D0 = np.zeros((2*L,)*2,dtype=complex)
