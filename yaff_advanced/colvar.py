''' Ruben Demuynck + Molden Acknowledgement + Yaff Acknowledgement'''
'''This is a non exhaustive list of collective variables, add your own  ... I did not test all cvs in the list'''

import numpy as np
from yaff import *
from molmod.ic import *
import h5py

class DumySys(object):
    def __init__(self,pos):
        self.pos=pos

class CollectiveVariable(object):
    def __init__(self,name,type,atoms=None,parameter=None):
        self.name=name
        self.type=type
        self.atoms=atoms
        self.parameter=parameter
    def get_value(self,system):
        raise NotImplementedError
    def get_deriv(self,system):
        raise NotImplementedError
    def get_force(self,g,gpos,vtens,system):
        raise NotImplementedError
    def post(self,h5file):
        f=h5py.File(h5file)
        POS=f['trajectory/pos'][:]
        cv=np.zeros(len(POS[:,0,0]))
        for i,p in enumerate(POS):
            cv[i]=self.get_value(DumySys(p))
        return cv

class Angle(CollectiveVariable):
    def __init__(self,name,atoms):
        if len(atoms)!=3:
            raise ValueError('Distance is defined by two atoms, Angle three atoms and dihedral four atoms')
        CollectiveVariable.__init__(self,name,'angle',atoms=atoms)
    def get_value(self,system):
        return bend_angle(np.array([system.pos[self.atoms[0]],system.pos[self.atoms[1]],system.pos[self.atoms[2]]]))[0]
    def get_deriv(self,system):
        return bend_angle(np.array([system.pos[self.atoms[0]],system.pos[self.atoms[1]],system.pos[self.atoms[2]]]),deriv=1)[1]
    def get_force(self,g,gpos,vtens,system):
        cvderiv=self.get_deriv(system)
        for i,a in enumerate(self.atoms):
            gpos[a,:]+=g*cvderiv[i,:]
        return gpos,vtens

class Dihedral(CollectiveVariable):
    def __init__(self,name,atoms):
        if len(atoms)!=4:
            raise ValueError('Distance is defined by two atoms, Angle three atoms and dihedral four atoms')
        CollectiveVariable.__init__(self,name,'angle',atoms=atoms)
    def get_value(self,system):
        return dihed_angle(np.array([system.pos[self.atoms[0]],system.pos[self.atoms[1]],system.pos[self.atoms[2]],system.pos[self.atoms[3]]]))[0]
    def get_deriv(self,system):
        return dihed_angle(np.array([system.pos[self.atoms[0]],system.pos[self.atoms[1]],system.pos[self.atoms[2]],system.pos[self.atoms[3]]]),deriv=1)[1]
    def get_force(self,g,gpos,vtens,system):
        cvderiv=self.get_deriv(system)
        for i,a in enumerate(self.atoms):
            gpos[a,:]+=g*cvderiv[i,:]
        return gpos,vtens

class AngleCAU(CollectiveVariable):
    def __init__(self,name,atoms):
        if len(atoms)!=3:
            raise ValueError('Distance is defined by two atoms, Angle three atoms and dihedral four atoms')
        CollectiveVariable.__init__(self,name,'angle',atoms=atoms)
    def intro_ghost(self,system):
        ghost=np.zeros((3,3))
        for j,l in enumerate(self.atoms):
            if l>100 and l<200:
                ghost[j,:]=system.pos[l%100,:]-system.cell.rvecs[1,:]
            elif l>200:
                ghost[j,:]=system.pos[l%100,:]-system.cell.rvecs[1,:]-system.cell.rvecs[0,:]
            else:
                ghost[j,:]=system.pos[l,:]
        return ghost
    def get_value(self,system):
        ghost=self.intro_ghost(system)
        return bend_angle(np.array([ghost[0,:],ghost[1,:],ghost[2,:]]))[0]
    def get_deriv(self,system):
        ghost=self.intro_ghost(system)
        return bend_angle(np.array([ghost[0,:],ghost[1,:],ghost[2,:]]),deriv=1)[1]
    def get_force(self,g,gpos,vtens,system):
        cvderiv=self.get_deriv(system)
        for i,a in enumerate(self.atoms):
            gpos[a%100,:]+=g*cvderiv[i,:]
        return gpos,vtens

class Distance(CollectiveVariable):
    def __init__(self,name,atoms):
        if len(atoms)!=2:
            raise ValueError('Distance is defined by two atoms, Angle three atoms and dihedral four atoms')
        CollectiveVariable.__init__(self,name,'distance',atoms=atoms)
    def get_value(self,system):
        return bond_length(np.array([system.pos[self.atoms[0]],system.pos[self.atoms[1]]]))[0]
    def get_deriv(self,system):
        return bond_length(np.array([system.pos[self.atoms[0]],system.pos[self.atoms[1]]]),deriv=1)[1]
    def get_force(self,g,gpos,vtens,system):
        cvderiv=self.get_deriv(system)
        for i,a in enumerate(self.atoms):
            gpos[a,:]+=g*cvderiv[i,:]
        return gpos,vtens

class Volume(CollectiveVariable):
    def __init__(self):
        CollectiveVariable.__init__(self,'volume','volume')
    def get_value(self,system):
        return system.cell.volume
    def get_force(self,g,gpos,vtens,system):
        vtens+=np.identity(3)*g*self.get_value(system)
        return gpos,vtens

class CellParameter(CollectiveVariable):
    def __init__(self,name,parameter):
        if len(parameter)!=2:
            raise ValueError('Two indexes for the cell parameters required')
        CollectiveVariable.__init__(self,name,'cell',parameter=parameter)
    def get_value(self,system):
        return system.cell.rvecs[self.parameter[0],self.parameter[1]]
    def get_force(self,g,gpos,vtens,system):
        gvec=np.zeros((3,3))
        gvec[self.parameter[0],self.parameter[1]]=g
        vtens+=np.dot(system.cell.rvecs.T,gvec)
        return gpos,vtens

class Average(CollectiveVariable):
    def __init__(self,name,CVlist):
        if CVlist is None:
            raise ValueError('Give it another try! Add those collective variables over which you want to average')
        self.CVlist=CVlist
        CollectiveVariable.__init__(self,name,'average')

    def get_value(self,system):
        N=float(len(self.CVlist))
        v=0.
        for cv in self.CVlist:
            v+=cv.get_value(system)
        return 1./N*v

    def get_force(self,g,gpos,vtens,system):
        N=float(len(self.CVlist))
        for cv in self.CVlist:
            gpos,vtens=cv.get_force(1./N*g,gpos,vtens,system)
        return gpos,vtens
