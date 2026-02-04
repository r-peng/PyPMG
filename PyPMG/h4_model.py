import numpy as np
import scipy,itertools
import torch
from PyPMG.pmg import * 
from PyPMG.vmc import *
class H4MinimalState(PMGState):
    def __init__(self,x,**kwargs):
        nsites = 4,4
        nelec = 2,2
        super().__init__(nsites,nelec,**kwargs)

        def fxn(cf):
            return (-1)**cf[1]
        self.add_pmg(5,7,fxn)

        def fxn(cf):
            return (-1)**cf[4]
        self.add_pmg(0,2,fxn)

        def fxn(cf):
            return (-1)**cf[5]
        self.add_pmg(0,2,fxn)

        self._update(x)
        #if RANK==0:
        #    idx = 23,8,21,10,1,15,26,3,16,19,27,14,11,0,24
        ##    idx = 1,9,8,2,0,7,5
        #    for ix in idx:
        #        print(ix,self.mg.pairs[ix])
        #exit()
