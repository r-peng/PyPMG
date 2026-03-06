import numpy as np
import scipy,itertools
from PyPMG.pmg import * 
def get_h8_minimum(HF_typ,pmg_typ,manual_derivative=True,remove_redundant=False,**kwargs):
    nsites = 8,8
    nelec = 4,4
    if manual_derivative:
        psi = PMGState_manual(nsites,nelec,**kwargs)
    else:
        psi = PMGState_autodiff(nsites,nelec,**kwargs)

    if pmg_typ==1:
        decimated = 0,3,4,7,8,11,12,15
        hop_ls = complement(psi.nsite,decimated)
        hop_ls = get_hop_ls(hop_ls,HF_typ)
        jac_by = 'frechet'
        psi.add_pmg(hop_ls,decimated,jac_by=jac_by)

        decimated = 3,7,11,15
        hop_ls = complement(psi.nsite,decimated)
        hop_ls = get_hop_ls(hop_ls,HF_typ)
        jac_by = 'frechet'
        psi.add_pmg(hop_ls,decimated,jac_by=jac_by)

        remove_unphysical = None 
        jac_by = 'ad'
        psi.add_mg(HF_typ,remove_redundant=remove_redundant,remove_unphysical=decimated,jac_by=jac_by)
    if pmg_typ==2:
        decimated = 0,2,4,6,8,10,12,14
        hop_ls = complement(psi.nsite,decimated)
        hop_ls = get_hop_ls(hop_ls,HF_typ)
        jac_by = 'frechet'
        psi.add_pmg(hop_ls,decimated,jac_by=jac_by)

        #decimated = 3,7,11,15
        #hop_ls = complement(psi.nsite,decimated)
        #hop_ls = get_hop_ls(hop_ls,HF_typ)
        #jac_by = 'frechet'
        #psi.add_pmg(hop_ls,decimated,jac_by=jac_by)

        remove_unphysical = None 
        jac_by = 'ad'
        psi.add_mg(HF_typ,remove_redundant=remove_redundant,remove_unphysical=decimated,jac_by=jac_by)
    else:
        raise ValueError
    psi.get_nparam()
    return psi
