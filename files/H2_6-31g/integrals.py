import pyscf
from pyscf import gto
import numpy as np
from pyscf import ao2mo
import h5py
np.set_printoptions(suppress=True,precision=5)

r_list = [1.0,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.1,1.11,1.12,1.13,1.14,1.15,1.16,1.17,1.18,1.19,1.2,1.21,1.22,1.23,1.24,1.25,1.26,1.27,1.28,1.29,1.3,1.31,1.32,1.33,1.34,1.35,1.36,1.37,1.38,1.39,1.4,1.41,1.42,1.43,1.44,1.45,1.46,1.47,1.48,1.49,1.5,1.51,1.52,1.53,1.54,1.55,1.56,1.57,1.58,1.59,1.6,1.61,1.62,1.63,1.64,1.65,1.66,1.67,1.68,1.69,1.7,1.71,1.72,1.73,1.74,1.75,1.76,1.77,1.78,1.79,1.8,1.81,1.82,1.83,1.84,1.85,1.86,1.87,1.88,1.89,1.9,1.91,1.92,1.93,1.94,1.95,1.96,1.97,1.98,1.99,2.0]

def get_orthoAO(S, LINDEP_CUTOFF=1e-14):
    sdiag, Us = np.linalg.eigh(S)
    X = Us[:, sdiag > LINDEP_CUTOFF] / np.sqrt(sdiag[sdiag > LINDEP_CUTOFF])
    return X

for r in r_list:
    print(f'r={r}')
    mol = gto.Mole()
    #mol.atom = f'''
    #H 0.0 0.0 0.0
    #H 0.0 0.0 1.23
    #H {r} 0.0 0.0
    #H {r} 0.0 1.23
    #'''
    mol.atom = f'''
    H 0.0 0.0 0.0
    H {r} 0.0 0.0
    '''

    mol.basis = '6-31g'
    mol.spin = 0
    mol.charge = 0
    mol.build()

    # oao integrals for the molecule
    S = mol.intor('int1e_ovlp')
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    hcore = T + V
    eri = mol.intor('int2e')
    ecore = mol.energy_nuc()
    X = get_orthoAO(S)
    xinv = np.linalg.inv(X)
    # np.testing.assert_allclose(X.T @ S @ X, np.eye(X.shape[1]), atol=1e-10)

    hcore_oao = X.T.conj() @ hcore @ X
    n_oao = X.shape[1]
    print(n_oao)
    exit()
    eri_packed = ao2mo.kernel(mol, X)
    eri_oao = ao2mo.restore(1, eri_packed, n_oao)

    #mf = pyscf.scf.RHF(mol)
    #mf.kernel()

    #moa = mf.mo_coeff
    #moa_oao = xinv @ moa
    #nocca, noccb = mol.nelec

    #dma_oao = np.einsum("mn, pn -> mp", moa_oao[:, :nocca], moa_oao[:, :nocca].conj(), optimize=True)

    #E1_oao = 2. * np.einsum("mn, nm ->", hcore_oao, dma_oao, optimize=True) 
    #EJ_oao = 2. * np.einsum("mnls, nm, sl->", eri_oao, dma_oao, dma_oao, optimize=True)

    #EK_oao_a = - np.einsum("mnls,nl,sm->", eri_oao, dma_oao, dma_oao, optimize=True) 
    #EHF_oao =  E1_oao + EJ_oao + EK_oao_a + ecore
    #print(f"HF energy in OAO basis: {EHF_oao}")
    #print(f"1-body energy in OAO basis: {E1_oao}")
    #print(f"Coulomb energy in OAO basis: {EJ_oao}")
    #print(f"alpha Exchange energy in OAO basis: {EK_oao_a}")

    with h5py.File(f'6-31g/h2_{r:.2f}.h5', 'w') as f:
        f.create_dataset('hcore_oao', data=hcore_oao)
        f.create_dataset('eri_oao', data=eri_oao)
        f.create_dataset('ecore', data=ecore)
