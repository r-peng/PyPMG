import numpy as np
import scipy,itertools
import torch
#from PyPMG.pmg import * 
from PyPMG.pmg_ import * 
from PyPMG.vmc import *
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
class H4MinimalState(PMGState):
    def __init__(self,HF_typ,pmg_typ=1,remove_redundant=False,**kwargs):
        nsites = 4,4
        nelec = 2,2
        super().__init__(nsites,nelec,**kwargs)

        remove_unphysical = None
        if pmg_typ==1:
            decimated = 1,3,4,6,5,7

            hop_ls = (0,2), 
            self.add_pmg(hop_ls,ctr_ls=decimated)

            hop_ls = (5,7),
            ctr_ls = 1,3,4,6,
            self.add_pmg(hop_ls,ctr_ls=ctr_ls)
        elif pmg_typ==(2,1):
            decimated = 1,3,5,6,7,

            hop_ls = (0,2),(0,4),(2,4) 
            self.add_pmg(hop_ls,ctr_ls=decimated)

            ctr_ls = 1,6,
            hop_ls = complement(self.nsite,ctr_ls)
            hop_ls = get_hop_ls(hop_ls,HF_typ)
            self.add_pmg(hop_ls,ctr_ls=ctr_ls)
        elif pmg_typ==(2,2):
            decimated = 1,3,5,6

            hop_ls = (0,2),(0,4),(2,4)
            self.add_pmg(hop_ls,ctr_ls=decimated)

            ctr_ls = 1,6,
            hop_ls = complement(self.nsite,ctr_ls)
            hop_ls = get_hop_ls(hop_ls,HF_typ)
            self.add_pmg(hop_ls,ctr_ls=ctr_ls,remove_redundant=remove_redundant)

            remove_unphysical = ctr_ls 
        elif pmg_typ==3:
            jac_by = 'frechet'
            jac_by = 'ad'
            decimated = 1,3,5,6,

            ctr_ls = decimated 
            hop_ls = complement(self.nsite,ctr_ls)
            hop_ls = get_hop_ls(hop_ls,HF_typ)
            #self.add_pmg(hop_ls,ctr_ls=ctr_ls)
            self.add_pmg(hop_ls,decimated,jac_by=jac_by)

            ctr_ls = 1,6,
            decimated = 1,6,
            hop_ls = complement(self.nsite,ctr_ls)
            hop_ls = get_hop_ls(hop_ls,HF_typ)
            #self.add_pmg(hop_ls,ctr_ls=ctr_ls,remove_redundant=remove_redundant)
            self.add_pmg(hop_ls,decimated,remove_redundant=remove_redundant,jac_by=jac_by)

            remove_unphysical = ctr_ls 
        else:
            raise ValueError
        self.add_mg(HF_typ,remove_redundant=remove_redundant,remove_unphysical=remove_unphysical,jac_by=jac_by)
        self.get_nparam()
        #exit()
class H4MinimalHF(PMGState):
    def __init__(self,HF_typ,pmg_typ,**kwargs):
        nsites = 4,4
        nelec = 2,2
        super().__init__(nsites,nelec,**kwargs)
        def remove(hop_ls):
            ls = [(0,2),(4,6),(1,3),(5,7)]
            for pair in ls:
                if pair in hop_ls:
                    hop_ls.remove(pair)
            return hop_ls

        remove_unphysical = None
        if pmg_typ==1:
            hop_ls = (2,4), 
            ctr_ls = 0,1,3,5,6,7,
            self.add_pmg(hop_ls,ctr_ls=ctr_ls)

            hop_ls = (3,5),
            ctr_ls = 0,1,6,7,
            self.add_pmg(hop_ls,ctr_ls=ctr_ls)
        elif pmg_typ==2:
            #ctr_ls = 1,3,4,6,
            ctr_ls = 1,3,5,6,
            hop_ls = complement(self.nsite,ctr_ls)
            hop_ls = get_hop_ls(hop,HF_typ)
            hop_ls = remove(hop_ls)
            self.add_pmg(hop_ls,ctr_ls=ctr_ls)

            ctr_ls = 1,6,
            hop_ls = complement(self.nsite,ctr_ls)
            hop_ls = get_hop_ls(hop_ls,HF_typ)
            hop_ls = remove(hop_ls)
            self.add_pmg(hop_ls,ctr_ls=ctr_ls)

            remove_unphysical = ctr_ls 
        else:
            raise ValueError

        self.add_mg(HF_typ,remove_redundant=remove_redundant,remove_unphysical=remove_unphysical)
        self.get_nparam()
class H4_6_31G(PMGState):
    def __init__(self,HF_typ,pmg_typ=1,remove_redundant=False,**kwargs):
        nsites = 8,8
        nelec = 2,2
        super().__init__(nsites,nelec,**kwargs)

        remove_unphysical = None
        jac_by = 'frechet'
        if pmg_typ==1:
            decimated = 1,3,5,7,8,10,12,14

            ctr_ls = decimated 
            hop_ls = complement(self.nsite,decimated)
            hop_ls = get_hop_ls(hop_ls,HF_typ)
            self.add_pmg(hop_ls,decimated,jac_by=jac_by)
        else:
            raise ValueError

        self.add_mg(HF_typ,remove_redundant=remove_redundant,remove_unphysical=remove_unphysical,jac_by=jac_by)
        self.get_nparam()
