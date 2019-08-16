'''Ruben Demuynck + Yaff acknowledgement'''

from yaff import *
import numpy as np
from molmod.unit_cells import UnitCell
from molmod.ic import bend_angle
class ForcePartMTD(ForcePart):
    def __init__(self,system,hill,steps):
        self.name='MTD'
        self.hill=hill
        self.system=system
        self.B=np.zeros((steps,len(hill.CV)))
        self.H=np.zeros(steps)
        ForcePart.__init__(self,self.name,system)

    def determineF(self,system):
        N=system.natom
        self.localGPos=np.zeros((N,3))
        self.localVTens=np.zeros((3,3))
        cv=[cv.get_value(system) for cv in self.hill.CV]
        kernel=1
        for i,value in enumerate(self.hill.CV):
            kernel*=np.exp(-(cv[i]-self.B[:,i])**2/2./self.hill.w[i]**2)
        for i,value in enumerate(self.hill.CV):
            g=-np.sum(self.H[:]*self.hill.h*(cv[i]-self.B[:,i])/self.hill.w[i]**2*kernel)
            self.localGPos,self.localVTens=value.get_force(g,self.localGPos,self.localVTens,system)
        self.U=np.sum(self.H[:]*self.hill.h*kernel)
        return self.U

    def _internal_compute(self, gpos, vtens):
        self.determineF(self.system)
        if vtens is not None:
            vtens+=self.localVTens
        if gpos is not None:
            gpos+=self.localGPos
        return self.U

class MTDHook(VerletHook):
    def __init__(self,hill,steps):
        self.counter=0
        self.updateT=steps
        self.updateC=0
        self.mem=np.zeros(len(hill.CV))
        self.hill=hill
        VerletHook.__init__(self)

    def init(self,iterative):
        pass

    def pre(self,iterative):
        pass

    def post(self,iterative):
        #print self.mem
        #print [cv.get_value(iterative.ff.system) for cv in self.hill.CV]
        self.mem+=np.array([cv.get_value(iterative.ff.system) for cv in self.hill.CV])
        self.counter+=1
        if self.counter%self.updateT==0:
            self.hill.ffPart.B[self.updateC,:]=self.mem/self.counter
            self.hill.ffPart.H[self.updateC]=1
            self.updateC+=1
            self.counter=0
            self.mem=0



class Hills(object):
    def __init__(self,collectiveVariable, width=None, height=None):
        '''collectiveVariable
                        an object from the CollectiveVariable class
           atoms
                in the latter three cases, one should identify the distance, angle or dihedral using atom numbers (PYTHON STYLE starting at zero)
           width/height
                make a guess for both the hill height and width minimizing both bias and variance of the free energy profile.
        '''
        if isinstance(collectiveVariable, list):
            self.CV=collectiveVariable
            self.w=width
        else:
            self.CV=[]
            self.w=[]
            self.CV.append(collectiveVariable)
            self.w.append(width)
        if width is None:
            '''In the near future we have to add the variational approach towards metadynamics'''
            raise NotImplementedError
        self.w=width
        self.h=height
        self.ffPart=None
        self.Hook=None

class HillsState(StateItem):
    def __init__(self,hill):
        self.hill=hill
        StateItem.__init__(self,'mtd')

    def get_value(self,iterative):
        return self.hill.ffPart.B

class Metadynamics1D(VerletIntegrator):
    def __init__(self,ff,timestep,MetaSteps,MDSteps,Hills,state=None,hooks=None,velo0=None,temp0=300,scalevel0=True,time0=0.0,ndof=None,counter0=0):
        self.steps=MDSteps*MetaSteps
        #print self.steps
        state= [] if state is None else state
        for hill in Hills:
            state.append(HillsState(hill))
            hill.ffPart=ForcePartMTD(ff.system,hill,MetaSteps)
            ff.add_part(hill.ffPart)
            hill.Hook=MTDHook(hill,MDSteps)
            hooks.append(hill.Hook)
        VerletIntegrator.__init__(self, ff, timestep, state=state, hooks=hooks, vel0=velo0,temp0=temp0, scalevel0=scalevel0, time0=time0, ndof=ndof, counter0=counter0)

    def runMeta(self):
        #print self.steps
        self.run(self.steps)
