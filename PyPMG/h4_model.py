import numpy as np
import scipy,itertools
from PyPMG.pmg import * 
def get_h4_minimum(HF_typ,manual_derivative=False,jac_by=None,remove_redundant=False,**kwargs):
    nsites = 4,4
    nelec = 2,2
    if manual_derivative:
        psi = PMGState_manual(nsites,nelec,**kwargs)
    else:
        psi = PMGState_autodiff(nsites,nelec,**kwargs)

    decimated = 1,3,5,6
    hop_ls = (0,2),(0,4),(2,4)
    psi.add_pmg(hop_ls,decimated,jac_by=jac_by)

    decimated = 1,6,
    hop_ls = complement(psi.nsite,decimated)
    hop_ls = get_hop_ls(hop_ls,HF_typ)
    psi.add_pmg(hop_ls,decimated,remove_redundant=remove_redundant,jac_by=jac_by)

    psi.add_pmg(HF_typ,None,remove_unphysical=decimated,jac_by=jac_by)
    psi.get_nparam()
    return psi
