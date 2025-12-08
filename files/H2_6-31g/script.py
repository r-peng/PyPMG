import numpy as np
import h5py,itertools
np.set_printoptions(precision=6,suppress=True)

for R in np.arange(1,2.01,.01):
    f = h5py.File(f'6-31g/h2_{R:.2f}.h5','r')
    const = f['ecore'][()]
    eri = f['eri_oao'][:] 
    hcore = f['hcore_oao'][:]
    f.close()

    # check symmetry in chemistry notation (b1,k1,b2,k2)
    assert np.linalg.norm(eri-eri.transpose(1,0,2,3))<1e-12
    assert np.linalg.norm(eri-eri.transpose(0,1,3,2))<1e-12
    assert np.linalg.norm(eri-eri.transpose(2,3,0,1))<1e-12
    print(f'R={R:.2f},ecore={const},nsite={hcore.shape[0]}')
    #print('hcore:')
    #for i,j in itertools.product(range(2),repeat=2):
    #    if np.fabs(hcore[i,j])>1e-10:
    #        print(i,j,hcore[i,j])
    #print('eri:')
    #for i,j,k,l in itertools.product(range(2),repeat=4):
    #    if np.fabs(eri[i,j,k,l])>1e-10:
    #        print(i,j,k,l,eri[i,j,k,l])
    #continue

    # permuting into spin-orbital 
    # H = \sum_{pq}h_{pq}a_p^\dagger a_q
    #   + \frac{1}{2}\sum_{pqrs}v_{pqrs}a_p^\dagger a_q^\dagger a_s a_r
    #   + const
    # p,q,r,s are spin-orbital indices, 
    # even/odd indices are up/down-spin
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
    v /= 2
