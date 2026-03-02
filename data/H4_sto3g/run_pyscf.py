import numpy as np
from pyscf import gto,scf,fci
from pyscf.fci import cistring
import scipy
np.set_printoptions(suppress=True,precision=6)

norb = 4
nelec = (2, 2)
na = cistring.num_strings(norb, nelec[0])  # 6
nb = cistring.num_strings(norb, nelec[1])  # 6
def bits_to_occ(bitstr, norb):
    return [p for p in range(norb) if (bitstr >> p) & 1]
def det_from_indices(norb, nelec, ia, ib):
    nalpha, nbeta = nelec
    a_bits = cistring.addr2str(norb, nalpha, ia)
    b_bits = cistring.addr2str(norb, nbeta, ib)
    return bits_to_occ(a_bits, norb), bits_to_occ(b_bits, norb)



dR = 0.02
Rmin = 1.01
nR = 50 
R = Rmin
for i in range(nR):
    print()
    print(f'######################## R={R} ###########################')
    
    print('building molecule...')
    mol = gto.Mole()
    mol.atom = [['H',(0,0,0)],
                ['H',(R,0,0)],
                ['H',(0,0,1.23)],
                ['H',(R,0,1.23)]] 
    mol.basis = 'sto-3g'
    mol.build()
    print('ecore=',mol.enuc)
    print()
    print('nelec=',mol.nelec)
    
    print('running unrestricted Hartree Fock...')
    mf = scf.UHF(mol)
    dm = mf.init_guess_by_minao()
    mf.kernel(dm0=dm)
    print('Hartree Fock total energy=',mf.e_tot) # electron + nuclear
    print('Hartree Fock electronic energy=',mf.e_tot-mol.enuc) # electron + nuclear
    print('mo energies=',mf.mo_energy)
    print(mf.mo_coeff)
    S = mol.intor('int1e_ovlp')
    sdiag, Us = np.linalg.eigh(S)
    LINDEP_CUTOFF = 1e-6
    Us = Us[:, sdiag > LINDEP_CUTOFF]
    sdiag = sdiag[sdiag > LINDEP_CUTOFF]
    X = np.dot(Us/np.sqrt(sdiag),Us.T)
    Xinv = np.linalg.inv(X)
    moa = np.dot(Xinv,mf.mo_coeff[0])
    print('moa')
    print(moa)
    
    print('running unrestricted FCI...')
    cisolver = fci.FCI(mf)
    e_fci = cisolver.kernel()[0]
    print('FCI total energy=',e_fci)
    print('FCI electronic energy=',e_fci-mol.enuc)
    print('FCI correlation energy=',e_fci-mf.e_tot)
    #print(cisolver.ci.size)
    #ci = cisolver.ci
    #ci2 = ci.reshape(na, nb)  # ci is your length-36 vector
    #for k in range(na * nb):
    #    ia, ib = divmod(k, nb)
    #    occ_a, occ_b = det_from_indices(norb, nelec, ia, ib)
    #    if abs(ci[ia,ib])>1e-10:
    #        print(f"k={k:2d}  ia={ia} ib={ib}   occ_a={occ_a}  occ_b={occ_b}   c={ci2[ia,ib]: .6e}")

    R += dR
