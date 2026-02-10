import pyscf
from pyscf import gto,lo
import numpy as np
from pyscf import ao2mo
import h5py,itertools
np.set_printoptions(suppress=True,precision=6)

dR = 0.02
Rmin = 1.01
nR = 50
typ = 'lowdin'
#typ = 'diag'
#typ = 'hf_mo'

def get_orthoAO(mol, LINDEP_CUTOFF=1e-14):
    if typ=='hf_mo': 
        mf = pyscf.scf.RHF(mol)
        mf.kernel()
        return mf.mo_coeff
    S = mol.intor('int1e_ovlp')
    sdiag, Us = np.linalg.eigh(S)
    Us = Us[:, sdiag > LINDEP_CUTOFF]
    sdiag = sdiag[sdiag > LINDEP_CUTOFF]
    if typ=='lowdin':
        X = np.dot(Us/np.sqrt(sdiag),Us.T)
    elif typ=='diag':
        X = Us/np.sqrt(sdiag)
    else:
        raise NotImplementedError
    return X

for i in range(nR):
    r = Rmin+i*dR 
    print(f'r={r}')
    mol = gto.Mole()
    mol.atom = f'''
    H 0.0 0.0 0.0
    H {r} 0.0 0.0
    H 0.0 0.0 1.23
    H {r} 0.0 1.23
    '''

    mol.basis = 'sto-3g'
    mol.spin = 0
    mol.charge = 0
    mol.build()

    # oao integrals for the molecule
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    hcore = T + V
    eri = mol.intor('int2e')
    ecore = mol.energy_nuc()
    X = get_orthoAO(mol)
    # np.testing.assert_allclose(X.T @ S @ X, np.eye(X.shape[1]), atol=1e-10)

    hcore_oao = X.T.conj() @ hcore @ X
    n_oao = X.shape[1]
    eri_packed = ao2mo.kernel(mol, X)
    eri_oao = ao2mo.restore(1, eri_packed, n_oao)

    # moa_oao = xinv @ moa
    # nocca, noccb = h4_mol.nelec

    # dma_oao = np.einsum("mn, pn -> mp", moa_oao[:, :nocca], moa_oao[:, :nocca].conj(), optimize=True)

    # E1_oao = 2. * np.einsum("mn, nm ->", hcore_oao, dma_oao, optimize=True) 
    # EJ_oao = 2. * np.einsum("mnls, nm, sl->", eri_oao, dma_oao, dma_oao, optimize=True)

    # EK_oao_a = - np.einsum("mnls,nl,sm->", eri_oao, dma_oao, dma_oao, optimize=True) 
    # EHF_oao =  E1_oao + EJ_oao + EK_oao_a + ecore
    # print(f"HF energy in OAO basis: {EHF_oao}")
    # print(f"1-body energy in OAO basis: {E1_oao}")
    # print(f"Coulomb energy in OAO basis: {EJ_oao}")
    # print(f"alpha Exchange energy in OAO basis: {EK_oao_a}")

    with h5py.File(f'{typ}/h4_{r:.2f}.h5', 'w') as f:
        f.create_dataset('hcore_oao', data=hcore_oao)
        f.create_dataset('eri_oao', data=eri_oao)
        f.create_dataset('ecore', data=ecore)
    #exit()
