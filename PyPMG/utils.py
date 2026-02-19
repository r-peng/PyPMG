import numpy as np
def soft_inv(M,thresh=1e-6):
    u,s,v = np.linalg.svd(M,full_matrices=False)
    s = s[s>s[0]*thresh]
    l = len(s)
    u,v = u[:,:l],v[:l,:]
    return np.dot(v.T.conj()/s.reshape(1,l),u.T.conj())
def rdm12covariance(G):
    n = G.shape[0]
    I = np.eye(n)
    g = np.zeros((n*2,)*2,dtype=complex)
    g[::2,::2] = g[1::2,1::2] = I-G.T.conj()+G
    g[1::2,::2] = 1j*(I-G.T.conj()-G)
    g[::2,1::2] = -1j*(I-G.T.conj()-G)
    return np.eye(2*n)-g
def covariance2rdm1(g):
    g = np.eye(g.shape[0])-g
    #print(g[::2,::2])
    #print(g[1::2,1::2])
    #print(-1j*g[::2,1::2])
    #print(1j*g[1::2,::2])
    G = g[::2,::2]+g[1::2,1::2]-1j*g[::2,1::2]+1j*g[1::2,::2]
    return G/4
def covariance_product(g1,g2):
    I = np.eye(g1.shape[0])
    g12 = np.dot(g1,g2)

    w = np.linalg.eigvals(g12)
    idx = np.argsort(w.real)
    w = w[idx]
    tr = 1.
    for wi in w[::4]:
        a,b = wi.real,wi.imag
        tr *= (1+2*a+a**2+b**2)/4

    M = soft_inv(I+g12)
    return I-np.dot(I-g2,np.dot(M,I-g1)),tr
def partial_trace(C,E,decimated,active=None):
    nsite = C.shape[0]
    if active is None:
        active = list(set(range(nsite))-set(decimated))
        active.sort()
    idx = np.array(active+decimated)
    na = len(active)

    C = C[idx,:][:,idx] 
    A,B,C,D = C[:na,:na],C[:na,na:],C[na:,:na],C[na:,na:]
    M = D+E
    Minv = soft_inv(M)
    BMC = np.dot(B,np.dot(Minv,C))

    S = np.linalg.det(E)*np.linalg.det(M)/2**na
    return A-BMC,S
