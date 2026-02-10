import numpy as np
import scipy,itertools
import torch
from PyPMG.pmg import * 
from PyPMG.vmc import *
class H4MinimalState(PMGState):
    def __init__(self,x,**kwargs):
        nsites = 4,4
        nelec = 2,2

        pmg_ls = []

        nsite = sum(nsites)
        hop_ls = (0,2), 
        ctr_ls = (1,),(3,),(4,),(6,),(5,),(7,)
        pmg_ls.append(PMG(nsite,hop_ls,ctr_ls=ctr_ls))

        hop_ls = (5,7),
        ctr_ls = (1,),(3,),(4,),(6,),
        pmg_ls.append(PMG(nsite,hop_ls,ctr_ls=ctr_ls))

        hop_ls = 'UHF' 
        pmg_ls.append(MG(nsites,hop_ls=hop_ls))

        super().__init__(nsites,nelec,pmg_ls,**kwargs)
        assert len(x)==self.nparam
        self._update(x)

class H4MinimalHF(PMGState):
    def __init__(self,x,**kwargs):
        nsites = 4,4
        nelec = 2,2

        pmg_ls = []

        nsite = sum(nsites)
        hop_ls = (2,4), 
        ctr_ls = (0,),(1,),(3,),(5,),(6,),(7,)
        pmg_ls.append(PMG(nsite,hop_ls,ctr_ls=ctr_ls))

        hop_ls = (3,5),
        ctr_ls = (0,),(1,),(6,),(7,),
        pmg_ls.append(PMG(nsite,hop_ls,ctr_ls=ctr_ls))

        hop_ls = 'UHF' 
        pmg_ls.append(MG(nsites,hop_ls=hop_ls))

        super().__init__(nsites,nelec,pmg_ls,**kwargs)
        assert len(x)==self.nparam
        self._update(x)

class H4MinimalHF2(PMGState):
    def __init__(self,x,path,no_repeat=True,**kwargs):
        nsites = 4,4
        nelec = 2,2

        hop_ls = (0,2),(0,3),(1,2),(1,3)
        hop_ls = [(2*p+i,2*q+i) for p,q in hop_ls for i in (0,1)]
        ctr_ls = [0,1,6,7]
        G = PMGGraph() 
        layers = G.get_all_pmg(hop_ls,ctr_ls,iprint=0)
        layers = layers[::-1]

        pmg_ls = []
        nsite = sum(nsites)
        total_hops = set() 
        for p,layer in zip(path,layers):
            node = G.nodes[layer[p]]
            hop_ls = node['hop_ls_left']
            if no_repeat:
                hop_ls = set(hop_ls)
                hop_ls = hop_ls - total_hops
                total_hops |= hop_ls
                hop_ls = list(hop_ls)
            ctr_ls = node['ctr_ls']
            ctr_ls = [(p,) for p in ctr_ls]
            if RANK==0:
                print('hop_ls=',hop_ls,)
                print('ctr_ls=',ctr_ls)
            pmg_ls.append(PMG(nsite,hop_ls,ctr_ls=ctr_ls))
        
        hop_ls = 'UHF' 
        pmg_ls.append(MG(nsites,hop_ls=hop_ls))

        super().__init__(nsites,nelec,pmg_ls,**kwargs)
        if RANK==0:
            print('nparam=',self.nparam)
        assert len(x)==self.nparam
        self._update(x)
