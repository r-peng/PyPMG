import numpy as np
from pyscf import gto,scf,fci

dR = 0.11
Rmin = 1.01
nR = 12
R = Rmin
for i in range(nR):
    print()
    print(f'######################## R={R} ###########################')
    
    print('building molecule...')
    mol = gto.Mole()
    mol.atom = [['H',(0,0,0)],
                ['H',(0,0,1.23)],
                ['H',(R,0,0)],
                ['H',(R,0,1.23)]] 
    mol.basis = 'sto-3g'
    mol.build()
    print('ecore=',mol.enuc)
    print()
    
    print('running unrestricted Hartree Fock...')
    mf = scf.UHF(mol)
    mf.kernel()
    print('Hartree Fock total energy=',mf.e_tot) # electron + nuclear
    print()
    
    print('running unrestricted FCI...')
    cisolver = fci.FCI(mf)
    e_fci = cisolver.kernel()[0]
    print('FCI total energy=',e_fci)
    print('FCI correlation energy=',e_fci-mf.e_tot)

    R += dR
