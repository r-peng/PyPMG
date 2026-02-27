import numpy as np
import scipy,itertools
from PyPMG.pmg import * 
def complement(nsite,ls1):
    ls2 = list(set(range(nsite)) - set(ls1))
    ls2.sort()
    return ls2
def get_hop_ls(ls,HF_typ):
    hop_ls = list(itertools.combinations(ls,2))
    if HF_typ=='GHF':
        return hop_ls
    elif HF_typ=='UHF':
        ls = []
        for (p,q) in hop_ls:
            if p%2==q%2:
                ls.append((p,q))
        return ls
    else:
        raise ValueError
def get_h4_minimum(HF_typ,pmg_typ,manual_derivative=False,remove_redundant=False,**kwargs):
    nsites = 4,4
    nelec = 2,2
    if manual_derivative:
        psi = PMGState_manual(nsites,nelec,**kwargs)
    else:
        psi = PMGState_autodiff(nsites,nelec,**kwargs)

    if pmg_typ==1:
        decimated = 1,3,4,6,5,7
        hop_ls = (0,2), 
        psi.add_pmg(hop_ls,decimated)

        decimated = 1,3,4,6,
        hop_ls = (5,7),
        psi.add_pmg(hop_ls,decimated)
        
        psi.add_mg(HF_typ)
    elif pmg_typ==(2,1):
        decimated = 1,3,5,6,7,
        hop_ls = (0,2),(0,4),(2,4) 
        psi.add_pmg(hop_ls,decimated)

        decimated = 1,6,
        hop_ls = complement(psi.nsite,decimated)
        hop_ls = get_hop_ls(hop_ls,HF_typ)
        psi.add_pmg(hop_ls,decimated)

        psi.add_mg(HF_typ)
    elif pmg_typ==(2,2):
        decimated = 1,3,5,6
        hop_ls = (0,2),(0,4),(2,4)
        psi.add_pmg(hop_ls,decimated)

        decimated = 1,6,
        hop_ls = complement(psi.nsite,decimated)
        hop_ls = get_hop_ls(hop_ls,HF_typ)
        psi.add_pmg(hop_ls,decimated)

        psi.add_mg(HF_typ,remove_unphysical=decimated)
    elif pmg_typ==3:
        jac_by = 'frechet'
        jac_by = 'ad'
        decimated = 1,3,5,6,
        hop_ls = complement(psi.nsite,decimated)
        hop_ls = get_hop_ls(hop_ls,HF_typ)
        psi.add_pmg(hop_ls,decimated,jac_by=jac_by)

        decimated = 1,6,
        hop_ls = complement(psi.nsite,decimated)
        hop_ls = get_hop_ls(hop_ls,HF_typ)
        psi.add_pmg(hop_ls,decimated,remove_redundant=remove_redundant,jac_by=jac_by)

        remove_unphysical = decimated 
        psi.add_mg(HF_typ,remove_redundant=remove_redundant,remove_unphysical=decimated,jac_by=jac_by)
    else:
        raise ValueError
    psi.get_nparam()
    return psi
def get_h4_minimum_hf(HF_typ,pmg_typ,manual_derivative=False,remove_redundant=False,**kwargs):
    nsites = 4,4
    nelec = 2,2
    if manual_derivative:
        psi = PMGState_manual(nsites,nelec,**kwargs)
    else:
        psi = PMGState_autodiff(nsites,nelec,**kwargs)
    def remove(hop_ls):
        ls = [(0,2),(4,6),(1,3),(5,7)]
        for pair in ls:
            if pair in hop_ls:
                hop_ls.remove(pair)
        return hop_ls

    if pmg_typ==1:
        decimated = 0,1,3,5,6,7,
        hop_ls = (2,4), 
        psi.add_pmg(hop_ls,decimated)

        decimated = 0,1,6,7,
        hop_ls = (3,5),
        psi.add_pmg(hop_ls,decimated)
        
        psi.add_mg(HF_typ)
    elif pmg_typ==2:
        decimated = 1,3,5,6,
        hop_ls = complement(psi.nsite,decimated)
        hop_ls = get_hop_ls(hop,HF_typ)
        hop_ls = remove(hop_ls)
        psi.add_pmg(hop_ls,decimated)

        decimated = 1,6,
        hop_ls = complement(psi.nsite,decimated)
        hop_ls = get_hop_ls(hop_ls,HF_typ)
        hop_ls = remove(hop_ls)
        psi.add_pmg(hop_ls,decimated)

        psi.add_mg(HF_typ,remove_unphysical=decimated)
    else:
        raise ValueError
    psi.get_nparam()
    return psi
def get_h4_6_31g(HF_typ,pmg_typ,manual_derivative=True,remove_redundant=False,**kwargs):
    nsites = 8,8
    nelec = 2,2
    if manual_derivative:
        psi = PMGState_manual(nsites,nelec,**kwargs)
    else:
        psi = PMGState_autodiff(nsites,nelec,**kwargs)

    jac_by = 'frechet'
    if pmg_typ==1:
        decimated = 1,3,5,7,8,10,12,14
        hop_ls = complement(psi.nsite,decimated)
        hop_ls = get_hop_ls(hop_ls,HF_typ)
        psi.add_pmg(hop_ls,decimated,jac_by=jac_by)

        psi.add_mg(HF_typ,remove_redundant=remove_redundant,remove_unphysical=remove_unphysical,jac_by=jac_by)
    else:
        raise ValueError
    psi.get_nparam()
    return psi
