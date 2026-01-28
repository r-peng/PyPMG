import numpy as np
from pyscf import gto,scf,fci

for R in np.arange(1.,2.01,.01):
    print()
    print(f'######################## R={R:.2f} ###########################')
    
    print('building molecule...')
    mol = gto.Mole()
    #mol.atom = [['H',(0,0,0)],
    #            ['H',(0,0,1.23)],
    #            ['H',(R,0,0)],
    #            ['H',(R,0,1.23)]] 
    mol.atom = f'''
    H 0.0 0.0 0.0
    H {R} 0.0 0.0
    '''
    mol.basis = 'sto-3g'
    mol.build()
    print('ecore=',mol.enuc)
    print()
    print('nelec=',mol.nelec)
    
    print('running unrestricted Hartree Fock...')
    mf = scf.UHF(mol)
    dm = mf.init_guess_by_minao()
    #print(dm)
    mf.kernel(dm0=dm)
    print('Hartree Fock total energy=',mf.e_tot) # electron + nuclear
    print()
    
    print('running unrestricted FCI...')
    cisolver = fci.FCI(mf)
    e_fci = cisolver.kernel()[0]
    print('FCI total energy=',e_fci)
    print('FCI correlation energy=',e_fci-mf.e_tot)
