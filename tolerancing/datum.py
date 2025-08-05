
from typing import List, Dict, TYPE_CHECKING
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

from .geometry import Geometry, GeoCalculator

if TYPE_CHECKING:
    from .component import Component

geocalc=GeoCalculator()

class Datum:
    
    def __init__(self, geo:Geometry, dim=None):
        self.geo:Geometry=geo
        self.dim=None
        self.tol:Tolerance=None
        self.fixed=False
        self.reference=None
        self.parent=None
    
    def set_reference(self, reference:'Datum'):
        self.reference=reference
        self.geo.set_reference(reference.geo)
    
    def set_parent(self, component:'Component'):
        self.parent=component
        
    def set_tolerance(self, tolerance:'Tolerance'):
        self.tol=tolerance
        
    def is_defined(self):
        pass
    
    def offset_parallel(self,d:float, direction:np.ndarray)->'Datum':
        newgeo = self.geo.derive(offset=d*direction)
        return type(self)(newgeo, dim = d)
    
    def rotate(self, angle:np.ndarray):
        newgeo = self.geo.derive(orient=angle)
        return type(self)(newgeo, dim=angle)
    
    def __repr__(self):
        return f"Datum()"

class ToleranceType(Enum):
    DIMENSION=0
    FLATNESS=1
    STRAIGHTNESS=2
    CYLINDRICITY=3
    CIRCULARITY=4
    PERPENDICULARITY=5
    PARALLELISM=6
    ANGULARITY=7
    POSITION=8
    SURFACEPROFILE=9
    LINEPROFILE=10
    TOTALRUNOUT=11
    CIRCULARRUNOUT=12
    CONCENTRICITY=13
    SYMMETRY=14
    

class Tolerance:
    
    def __init__(self, toltype:ToleranceType, value:float, negvalue:float=0):
        
        self.toltype=toltype
        self.pos=value
        self.neg=negvalue
    
    
    
class Relation(ABC):
    
    def __init__(self, offset=[0,0,0], orient=[0,0,1]):
        self.offset=np.array(offset)
        self.orient=np.array(orient)
    
    @abstractmethod
    def calculate(self, reference:Geometry, referent:Geometry)->None:
        pass
    
    def is_valid(self, reference:Geometry, referent:Geometry)->bool:
        pass
    
    def __call__(self, reference:Geometry, referent:Geometry):
        return self.calculate(reference, referent)

class Coincident(Relation):
    
    def calculate(self, reference:Geometry, referent:Geometry):
        referent.set_origin(reference.origin)
        referent.set_frame(reference.frame)
        
    def is_valid(self, reference:Geometry, referent:Geometry)->bool:
        return True

class Parallel(Relation):
    
    def calculate(self, reference:Geometry, referent:Geometry):
        referent.set_frame(reference.frame)
        
    def is_valid(self, reference:Geometry, referent:Geometry)->bool:
        return True
    
class Tangent(Relation):
    
    def calculate(self, reference:Geometry, referent:Geometry):
        pass
    
    def is_valid(self, reference:Geometry, referent:Geometry):
        
        pass
        