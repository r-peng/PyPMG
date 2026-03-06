import numpy as np
import scipy,itertools
from PyPMG.pmg import * 
def get_h8_minimum(HF_typ,pmg_typ,nlayer=1,manual_derivative=True,remove_redundant=False,**kwargs):
    nsites = 8,8
    nelec = 4,4
    if manual_derivative:
        psi = PMGState_manual(nsites,nelec,**kwargs)
    else:
        psi = PMGState_autodiff(nsites,nelec,**kwargs)

    if pmg_typ==1:
        decimated = 0,3,4,7,8,11,12,15
        order = 2
        hop_ls = complement(psi.nsite,decimated)
        hop_ls = get_hop_ls(hop_ls,HF_typ)
        jac_by = 'frechet'
        psi.add_pmg(hop_ls,decimated,jac_by=jac_by,order=order)

        if nlayer>1:
            decimated = 3,7,11,15
            hop_ls = complement(psi.nsite,decimated)
            hop_ls = get_hop_ls(hop_ls,HF_typ)
            jac_by = 'frechet'
            psi.add_pmg(hop_ls,decimated,remove_redundant=remove_redundant,jac_by=jac_by)

        remove_unphysical = None 
        jac_by = 'frechet'
        psi.add_pmg(HF_typ,None,remove_unphysical=remove_unphysical,jac_by=jac_by)
    elif pmg_typ==2:
        decimated = 0,2,4,6,10,12,14
        hop_ls = complement(psi.nsite,decimated)
        hop_ls = get_hop_ls(hop_ls,HF_typ)
        jac_by = 'frechet'
        order = 2
        psi.add_pmg(hop_ls,decimated,jac_by=jac_by,order=order)

        if nlayer>1:
            decimated = 0,4,10,14 
            hop_ls = complement(psi.nsite,decimated)
            hop_ls = get_hop_ls(hop_ls,HF_typ)
            jac_by = 'frechet'
            psi.add_pmg(hop_ls,decimated,remove_redundant=remove_redundant,jac_by=jac_by)

        remove_unphysical = None 
        jac_by = 'frechet'
        psi.add_pmg(HF_typ,None,remove_unphysical=remove_unphysical,jac_by=jac_by)
    elif pmg_typ==3:
        decimated = [0,2,4,6,10,12,14]+[3,7,11,15]
        hop_ls = complement(psi.nsite,decimated)
        hop_ls = get_hop_ls(hop_ls,HF_typ)
        jac_by = 'frechet'
        order = 2 
        psi.add_pmg(hop_ls,decimated,jac_by=jac_by,order=order)

        decimated = 0,3,4,7,10,11,12,15
        order = 1 
        hop_ls = complement(psi.nsite,decimated)
        hop_ls = get_hop_ls(hop_ls,HF_typ)
        jac_by = 'frechet'
        psi.add_pmg(hop_ls,decimated,remove_redundant=remove_redundant,jac_by=jac_by,order=order)

        decimated = 0,4,10,12
        order = 1 
        hop_ls = complement(psi.nsite,decimated)
        hop_ls = get_hop_ls(hop_ls,HF_typ)
        jac_by = 'frechet'
        psi.add_pmg(hop_ls,decimated,remove_redundant=remove_redundant,jac_by=jac_by,order=order)

        remove_unphysical = decimated 
        jac_by = 'frechet'
        psi.add_pmg(HF_typ,None,remove_unphysical=remove_unphysical,jac_by=jac_by)

    elif pmg_typ==4:
        decimated = [0,2,4,6,10,12,14]+[3,7,11,15]
        hop_ls = complement(psi.nsite,decimated)
        hop_ls = get_hop_ls(hop_ls,HF_typ)
        jac_by = 'frechet'
        order = 2 
        psi.add_pmg(hop_ls,decimated,jac_by=jac_by,order=order)

        decimated = 0,3,4,7,10,11,12,15
        order = 1 
        hop_ls = complement(psi.nsite,decimated)
        hop_ls = get_hop_ls(hop_ls,HF_typ)
        jac_by = 'frechet'
        psi.add_pmg(hop_ls,decimated,remove_redundant=remove_redundant,jac_by=jac_by,order=order)

        decimated = 3,7,11,15 
        order = 1 
        hop_ls = complement(psi.nsite,decimated)
        hop_ls = get_hop_ls(hop_ls,HF_typ)
        jac_by = 'frechet'
        psi.add_pmg(hop_ls,decimated,remove_redundant=remove_redundant,jac_by=jac_by,order=order)

        remove_unphysical = None 
        jac_by = 'frechet'
        psi.add_pmg(HF_typ,None,remove_unphysical=remove_unphysical,jac_by=jac_by)

    else:
        raise ValueError
    psi.get_nparam()
    return psi
def get_h50_minimum(HF_typ,pmg_typ,nlayer=1,manual_derivative=True,remove_redundant=False,**kwargs):
    nsites = 50,50
    nelec = 25,25
    if manual_derivative:
        psi = PMGState_manual(nsites,nelec,**kwargs)
    else:
        psi = PMGState_autodiff(nsites,nelec,**kwargs)

    if pmg_typ==1:
        decimated = list(range(0,sum(nsites),2))
