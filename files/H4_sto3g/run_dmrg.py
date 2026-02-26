import numpy as np
import pickle,h5py,itertools
from pyblock3.algebra.mpe import MPE
from pyblock3.hamiltonian import Hamiltonian 
from pyblock3.fcidump import FCIDUMP

norb = 4
nelec = (2, 2)
def print_amplitudes(mps):
    orbs_a = itertools.combinations(range(norb),nelec[0])
    orbs_b = itertools.combinations(range(norb),nelec[1])
    for orb_a,orb_b in itertools.product(orbs_a,orbs_b):
        det = [0] * norb
        for p in orb_a:
            det[p] += 2
        for p in orb_b:
            det[p] += 1
        amp = mps.amplitude(det)
        if abs(amp)>1e-6:
            print(f'occ_a={orb_a},occ_b={orb_b},c={amp}')

dR = 0.11
Rmin = 1.01
nR = 12
R = Rmin
typ = 'lowdin'
nelec = 4
for i in range(nR):
    print(f'##################### R={R:.2f} #############################')
    f = h5py.File(f'{typ}/h4_{R:.2f}.h5','r')
    const = f['ecore'][()]
    eri = f['eri_oao'][:] 
    hcore = f['hcore_oao'][:]
    f.close()
    
    nao = hcore.shape[0]
    nso = nao * 2
    h = np.zeros((nso,nso))
    h[::2,::2] = h[1::2,1::2] = hcore
    
    eri = eri.transpose(0,2,1,3) # permute to physicist notation (b1,b2,k1,k2)
    v = np.zeros((nso,nso,nso,nso))
    v[::2,::2,::2,::2] = eri.copy()
    v[1::2,1::2,1::2,1::2] = eri.copy()
    v[::2,1::2,::2,1::2] = eri.copy()
    v[1::2,::2,1::2,::2] = eri.copy()
    
    v_asym = v-v.transpose(0,1,3,2)
    assert np.linalg.norm(v_asym+v_asym.transpose(1,0,2,3))<1e-10
    assert np.linalg.norm(v_asym+v_asym.transpose(0,1,3,2))<1e-10
    assert np.linalg.norm(v_asym-v_asym.transpose(1,0,3,2))<1e-10
    
    fcidump = FCIDUMP(pg='c1',n_sites=nao,n_elec=nelec,twos=0,ipg=0,orb_sym=[0]*nao)
    hamil = Hamiltonian(fcidump,flat=True)
    cutoff = 1e-9
    def generate_terms(n_sites,c,d):
        for i,j in itertools.product(range(n_sites),repeat=2):
            for s in (0,1):
                coeff = h[2*i+s,2*j+s]
                #coeff = hcore[i,j]
                if abs(coeff)>cutoff:
                    yield coeff*c[i,s]*d[j,s]
        for i,j,k,l in itertools.product(range(n_sites),repeat=4):
            for s1,s2 in itertools.product((0,1),repeat=2):
                coeff = v[2*i+s1,2*j+s2,2*k+s1,2*l+s2] / 2 
                #coeff = v_asym[2*i+s1,2*j+s2,2*k+s1,2*l+s2] / 4 
                #coeff = eri[i,j,k,l] #/ 2
                if abs(coeff)>cutoff:
                    yield coeff*c[i,s1]*c[j,s2]*d[l,s2]*d[k,s1]
    
    mpo = hamil.build_mpo(generate_terms,cutoff=1e-9)
    mpo, _ = mpo.compress(left=True, cutoff=1E-9, norm_cutoff=1E-9)
    
    bdims = [100]
    noises = [1e-4,1e-5,1e-6]
    davthrds = None
    mps = hamil.build_mps(bdims[0])
    iprint = 0
    dmrg = MPE(mps, mpo, mps).dmrg(bdims=bdims,noises=noises,dav_thrds=davthrds,iprint=iprint,n_sweeps=20,tol=1e-12)
    ener = dmrg.energies[-1]
    print("Electronic Energy=",ener)
    print("Total Energy=",ener+const)
    smps = mps.to_non_flat().to_sliceable()
    print_amplitudes(smps)
    R += dR
    exit()
