import pyscf
from pyscf import gto,lo
import numpy as np
from pyscf import ao2mo
import h5py,itertools
np.set_printoptions(suppress=True,precision=6)

dR = 0.11
Rmin = 1.01
nR = 12

def get_orthoAO(S, LINDEP_CUTOFF=1e-14):
    sdiag, Us = np.linalg.eigh(S)
    Us = Us[:, sdiag > LINDEP_CUTOFF]
    sdiag = sdiag[sdiag > LINDEP_CUTOFF]
    X = np.dot(Us/np.sqrt(sdiag),Us.T)
    #X = Us/np.sqrt(sdiag)
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
    S = mol.intor('int1e_ovlp')
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    hcore = T + V
    eri = mol.intor('int2e')
    ecore = mol.energy_nuc()
    X = get_orthoAO(S)
    print('hcore')
    print(hcore)
    print('S')
    print(S)
    print('X')
    print(X)
    xinv = np.linalg.inv(X)
    # np.testing.assert_allclose(X.T @ S @ X, np.eye(X.shape[1]), atol=1e-10)

    hcore_oao = X.T.conj() @ hcore @ X
    print('hcore_oao')
    print(hcore_oao)
    exit()
    n_oao = X.shape[1]
    eri_packed = ao2mo.kernel(mol, X)
    eri_oao = ao2mo.restore(1, eri_packed, n_oao)
    #for i,j,k,l in itertools.product((0,1),repeat=4):
    #    if np.fabs(eri_oao[i,j,k,l])>1e-6:
    #        print(i,j,k,l,eri_oao[i,j,k,l])
    #exit()

    # mf = pyscf.scf.RHF(h4_mol)
    # mf.kernel()

    # moa = mf.mo_coeff
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

    with h5py.File(f'lowdin/h2_{r:.2f}.h5', 'w') as f:
        f.create_dataset('hcore_oao', data=hcore_oao)
        f.create_dataset('eri_oao', data=eri_oao)
        f.create_dataset('ecore', data=ecore)
