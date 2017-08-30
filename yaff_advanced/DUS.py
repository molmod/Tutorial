'''Ruben Demuynck + Yaff acknowledgement'''

from yaff import *
import numpy as np
from molmod.unit_cells import UnitCell
from molmod.ic import bend_angle
class ForcePartUS(ForcePart):
    def __init__(self,system,umbrella):
        self.name='Umbrella'
        self.umbrella=umbrella
        self.system=system
        ForcePart.__init__(self,self.name,system)

    def determineF(self,system):
        N=system.natom
        self.localGPos=np.zeros((N,3))
        self.localVTens=np.zeros((3,3))
        cv=[cv.get_value(system) for cv in self.umbrella.CV]
        kernel=1
        self.U=0
        for i,value in enumerate(self.umbrella.CV):
            self.U+=self.umbrella.k[i]/2.*(cv[i]-self.umbrella.cv0[i])**2
            g=self.umbrella.k[i]*(cv[i]-self.umbrella.cv0[i])
            self.localGPos,self.localVTens=value.get_force(g,self.localGPos,self.localVTens,system)
        return self.U

    def _internal_compute(self, gpos, vtens):
        self.determineF(self.system)
        if vtens is not None:
            vtens+=self.localVTens
        if gpos is not None:
            gpos+=self.localGPos
        return self.U



class Umbrella(object):
    def __init__(self,collectiveVariable, kappa, cv0):
        self.CV=collectiveVariable
        self.k=kappa
        self.cv0=cv0
        self.ffpart=None


class US(VerletIntegrator):
    def __init__(self,ff,timestep,MDSteps,umbrella,state=None,hooks=None,velo0=None,temp0=300,scalevel0=True,time0=0.0,ndof=None,counter0=0):
        self.steps=MDSteps
        umbrella.ffPart=ForcePartUS(ff.system,umbrella)
        ff.add_part(umbrella.ffPart)
        VerletIntegrator.__init__(self, ff, timestep, state=state, hooks=hooks, vel0=velo0,temp0=temp0, scalevel0=scalevel0, time0=time0, ndof=ndof, counter0=counter0)

    def runU(self):
        self.run(self.steps)
