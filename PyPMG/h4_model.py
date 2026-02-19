import numpy as np
import scipy,itertools
import torch
from PyPMG.pmg import * 
from PyPMG.vmc import *
def get_ctr_ls(in_ls,order=1):
    out_ls = []
    for r in range(1,order+1):
        out_ls += list(itertools.combinations(in_ls,r))
    return out_ls
def get_hop_ls(ctr_ls,nsite,HF_typ):
    hops = list(set(range(nsite)) - set(ctr_ls))
    hops.sort()
    hop_ls = list(itertools.combinations(hops,2))
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
        nsite = sum(nsites)

        pmg_ls = []
        if pmg_typ==1:
            decimated = 1,3,4,6,5,7

            hop_ls = (0,2), 
            ctr_ls = 1,3,4,6,5,7
            ctr_ls = get_ctr_ls(ctr_ls)
            pmg_ls.append(PMG(nsite,hop_ls,ctr_ls=ctr_ls))

            hop_ls = (5,7),
            ctr_ls = 1,3,4,6,
            ctr_ls = get_ctr_ls(ctr_ls)
            pmg_ls.append(PMG(nsite,hop_ls,ctr_ls=ctr_ls))
        elif pmg_typ==(2,1):
            decimated = 1,3,5,6,7,

            ctr_ls = decimated 
            hop_ls = get_hop_ls(ctr_ls,nsite,HF_typ)
            ctr_ls = get_ctr_ls(ctr_ls)
            pmg_ls.append(PMG(nsite,hop_ls,ctr_ls=ctr_ls))

            ctr_ls = 1,6,
            hop_ls = get_hop_ls(ctr_ls,nsite,HF_typ)
            ctr_ls = get_ctr_ls(ctr_ls)
            pmg_ls.append(PMG(nsite,hop_ls,ctr_ls=ctr_ls))
        elif pmg_typ==(2,2):
            decimated = 1,3,5,6

            ctr_ls = decimated 
            hop_ls = (0,2),(0,4),(2,4)
            ctr_ls = get_ctr_ls(ctr_ls)
            pmg_ls.append(PMG(nsite,hop_ls,ctr_ls=ctr_ls))

            ctr_ls = 1,6,
            hop_ls = get_hop_ls(ctr_ls,nsite,HF_typ)
            if remove_redundant:
                hop_ls = list(set(hop_ls)-set(pmg_ls[-1].hop_ls))
            ctr_ls = get_ctr_ls(ctr_ls)
            pmg_ls.append(PMG(nsite,hop_ls,ctr_ls=ctr_ls))

            HF_typ = list(itertools.combinations(range(nsite),2))
            HF_typ.remove((1,6))
            if remove_redundant:
                HF_typ = list(set(HF_typ)-set(pmg_ls[-1].hop_ls)-set(pmg_ls[-2].hop_ls))
        elif pmg_typ==3:
            decimated = 1,3,5,6,

            ctr_ls = decimated 
            hop_ls = get_hop_ls(ctr_ls,nsite,HF_typ)
            ctr_ls = get_ctr_ls(ctr_ls)
            pmg_ls.append(PMG(nsite,hop_ls,ctr_ls=ctr_ls))

            ctr_ls = 1,6,
            hop_ls = get_hop_ls(ctr_ls,nsite,HF_typ)
            if remove_redundant:
                hop_ls = list(set(hop_ls)-set(pmg_ls[-1].hop_ls))
            ctr_ls = get_ctr_ls(ctr_ls)
            pmg_ls.append(PMG(nsite,hop_ls,ctr_ls=ctr_ls))

            HF_typ = list(itertools.combinations(range(nsite),2))
            HF_typ.remove((1,6))
            if remove_redundant:
                HF_typ = list(set(HF_typ)-set(pmg_ls[-1].hop_ls)-set(pmg_ls[-2].hop_ls))
        else:
            raise ValueError

        pmg_ls.append(MG(nsites,hop_ls=HF_typ))
        if RANK==0:
            for pmg in pmg_ls:
                print('nparam=',pmg.nparam)
                print(pmg.hop_ls)
        #exit()

        super().__init__(nsites,nelec,pmg_ls,**kwargs)
        self.decimated = list(decimated)
        self.decimated.sort()
class H4MinimalHF(PMGState):
    def __init__(self,HF_typ,pmg_typ,**kwargs):
        nsites = 4,4
        nelec = 2,2
        nsite = sum(nsites)
        def remove(hop_ls):
            ls = [(0,2),(4,6),(1,3),(5,7)]
            for pair in ls:
                if pair in hop_ls:
                    hop_ls.remove(pair)
            return hop_ls

        pmg_ls = []
        if pmg_typ==1:
            hop_ls = (2,4), 
            ctr_ls = (0,),(1,),(3,),(5,),(6,),(7,)
            pmg_ls.append(PMG(nsite,hop_ls,ctr_ls=ctr_ls))

            hop_ls = (3,5),
            ctr_ls = (0,),(1,),(6,),(7,),
            pmg_ls.append(PMG(nsite,hop_ls,ctr_ls=ctr_ls))
        elif pmg_typ==2:
            #ctr_ls = 1,3,4,6,
            ctr_ls = 1,3,5,6,
            hop_ls = get_hop_ls(ctr_ls,nsite,HF_typ)
            hop_ls = remove(hop_ls)
            ctr_ls = get_ctr_ls(ctr_ls)
            pmg_ls.append(PMG(nsite,hop_ls,ctr_ls=ctr_ls))

            ctr_ls = 1,6,
            hop_ls = get_hop_ls(ctr_ls,nsite,HF_typ)
            hop_ls = remove(hop_ls)
            ctr_ls = get_ctr_ls(ctr_ls)
            pmg_ls.append(PMG(nsite,hop_ls,ctr_ls=ctr_ls))

            HF_typ = list(itertools.combinations(range(nsite),2))
            HF_typ.remove((1,6))
            HF_typ = remove(HF_typ)
        else:
            raise ValueError

        pmg_ls.append(MG(nsites,hop_ls=HF_typ))
        if RANK==0:
            for pmg in pmg_ls:
                print('nparam=',pmg.nparam)
                print(pmg.hop_ls)

        super().__init__(nsites,nelec,pmg_ls,**kwargs)
