import numpy as np
import h5py,itertools

# geometry taken from J. Chem. Theory Comput. 2019, 15, 311-324
dR = 0.11
Rmin = 1.01
nR = 12
R = Rmin
for i in range(nR):
    f = h5py.File(f'h4_sto3g_{R:.2f}.h5','r')
    const = f['ecore'][()]
    eri = f['eri_oao'][:] 
    hcore = f['hcore_oao'][:]
    f.close()

    # check symmetry in chemistry notation (b1,k1,b2,k2)
    assert np.linalg.norm(eri-eri.transpose(1,0,2,3))<1e-12
    assert np.linalg.norm(eri-eri.transpose(0,1,3,2))<1e-12
    assert np.linalg.norm(eri-eri.transpose(2,3,0,1))<1e-12
    print(f'R={R},ecore={const}')

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
    v *= 1/2
    #print('0,0=',h[0,0])
    #print('7,7=',h[7,7])
    #print('0,1,1,0=',v[0,1,1,0])
    #print('0,1,0,1=',v[0,1,0,1])

    R += dR
