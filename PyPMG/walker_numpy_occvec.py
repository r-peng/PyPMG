import itertools
import numpy as np
def fermion_act(x,oix,typ):
    kill = {'cre':1,'des':0}[typ]
    if x[oix]==kill:
        return None,0
    sign = (-1)**sum(x[:oix])
    y = list(x)
    y[oix] = 1-x[oix] 
    return tuple(y),sign
def string_act(x,ops,order=-1):
    if order==-1:
        ops = ops[::-1]
    s = np.zeros(len(ops),dtype=int)
    y = list(x) 
    for i,(oix,typ) in enumerate(ops):
        y,s[i] = fermion_act(y,oix,typ)
        if y is None:
            return None,0
    return tuple(y),np.prod(s)
def get_all_configs_u1(nsites,nelecs):
    configs = []
    for cf in itertools.product((0,1),repeat=nsites):
        if sum(cf)!=nelecs:
            continue
        configs.append(tuple(cf))
    return configs
def get_all_configs_u11(nsites,nelecs):
    alpha = get_all_configs_u1(nsites[0],nelecs[0])
    beta = get_all_configs_u1(nsites[1],nelecs[1])
    configs = []
    for cfa,cfb in itertools.product(alpha,beta):
        cf = [None] * (len(cfa)+len(cfb))
        cf[::2] = cfa
        cf[1::2] = cfb
        configs.append(tuple(cf))
    return configs
def get_exc_list_u1(x,nsite,nexs=2):
    xarr = np.array(x) 
    occ = np.argwhere(xarr>0.5).flatten()
    vir = np.argwhere(xarr<0.5).flatten()
    new_cfs = []
    for nex in range(1,nexs+1):
        occ_n = list(itertools.combinations(occ,nex))
        vir_n = list(itertools.combinations(vir,nex))
        for oix,vix in itertools.product(occ_n,vir_n): 
            y = list(x)
            for i,a in zip(oix,vix):
                y[i] = 1-y[i]
                y[a] = 1-y[a]
            new_cfs.append(tuple(y))
    return new_cfs
def precompute_for_exc_ls(nsite):
    return nsite
def get_exc_list_u11(x,nsite,nexs=2):
    cfs_u1 = get_exc_list_u1(x,nsite,nexs=nexs)
    cfs_u11 = []
    for cf in cfs_u1:
        if sum(cf[::2])=sum(cf[1::2]):
            cfs_u11.append(cf)
    return cfs_u11
def get_swap_list(x):
    xa,xb = x[::2],x[1::2]
    xa_arr,xb_arr = np.array(xa),np.array(xb) 
    u = np.argwhere(xa_arr*(1-xb_arr)>0.5).flatten() # singly occ up
    v = np.argwhere(xb_arr*(1-xa_arr)>0.5).flatten() # singly occ down
    new_cfs = []
    for p,q in itertools.product(u,v):
        y = list(x)
        for i in (2*p,2*q+1,2*q,2*p+1):
            y[i] = 1-y[i]
        new_cfs.append(tuple(y))
    return new_cfs 
def get_random_config_u1(nsite,nelec,rng,occ=None):
    if occ is None:
        occ = rng.choice(nsite,size=nelec,replace=False)
    cf = [0] * nsite
    for i in occ:
        cf[i] = 1
    return tuple(cf)
def get_random_config_u11(nsites,nelecs,rng,occ=(None,None)):
    if occ[0] is None:
        occ[0] = rng.choice(nsites[0],size=nelecs[0],replace=False)
    if occ[1] is None:
        occ[1] = rng.choice(nsites[1],size=nelecs[1],replace=False)
    occ = [2*p for p in occ[0]]+[2*p+1 for p in occ[1]]
    return get_random_config_u1(None,None,None,occ=occ)
def check_symmetry(x,symmetry,nsite,nelec):
    assert len(x)==nsite
    if symmetry=='u1':
        if sum(x)!=sum(nelec):
            return False 
    if symmetry=='u11':
        if sum(x[::2])!=nelec[0]:
            return False
        if sum(x[1::2])!=nelec[1]:
            return False
    return True
def ctr_ls_to_masks(ctr_ls):
    return ctr_ls
def ctr_ls_to_occ(cf,ctr_ls):
    occ = [None] * len(ctr_ls)
    for i,rs in enumerate(ctr_ls):
        occ[i] = int(np.prod([cf[r] for r in rs]))
    return tuple(occ)
def get_occ_indices(cf):
    return np.argwhere(cf).flatten()
