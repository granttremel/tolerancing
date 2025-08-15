

from enum import Enum
from typing import List, Dict,Any, TYPE_CHECKING
from abc import ABC, abstractmethod
import numpy as np
from numpy import sin, cos, tan
from scipy.spatial import transform

from tolerancing.form import ChebyshevBasis, FourierBasis, CylindricalBasis, SphericalHarmonicBasis, ZernikeBasis, NullBasis

class GeometryType(Enum):
    POINT=0
    
    AXIS=1
    CIRCLE=2
    CURVE=9

    PLANE=10
    CYLINDER=11
    SPHERE=12
    SURFACE=99

    PRISM=100
    BALL=101
    VOLUME=999
    
    NULL=-1

basis_map = {
    GeometryType.NULL:NullBasis(),
    GeometryType.AXIS: ChebyshevBasis(),
    GeometryType.PLANE: ZernikeBasis(),
    GeometryType.CYLINDER: CylindricalBasis(),  # Composite
    GeometryType.SPHERE: SphericalHarmonicBasis(),
}

def dimension(geo:GeometryType):
    if geo.value < 1:
        return 0
    elif geo.value < 10:
        return 1
    elif geo.value < 100:
        return 2
    elif geo.value < 1000:
        return 3
    else:
        return -1

intersection_general={
    (1,1):0,      # axis-axis -> point
    (1,10):0,     # axis-plane -> point
    (1,11):1,     # axis-cylinder -> axis (coaxial) or None
    (1,12):0,     # axis-sphere -> point(s)
    (2,10):0,     # circle-plane -> point(s)
    (10,10):1,    # plane-plane -> axis
    (10,11):2,    # plane-cylinder -> circle(s)
    (10,12):2,    # plane-sphere -> circle
    (11,11):1,    # cylinder-cylinder -> axis (coaxial) or None
    (12,12):2,    # sphere-sphere -> circle
}

intersection_tangent={
    (10,11):1,    # plane-cylinder tangent -> axis
    (10,12):0,    # plane-sphere tangent -> point
    (11,12):0,    # cylinder-sphere tangent -> point(s)
}

class GeometryBase(ABC):
    
    def __init__(self, **params):
        """
        origin is center point, frame is 3x3 matrix converting xyz coordinate into uvw coordinate ABOUT origin (if w or v are degenerate, the columns will be zero)
        for a 1d geometric object, u should be parallel to the tangent vector and v, w normal
        for a 2d geometric object, u should be normal and v,w tangent
        for a 3d geometric object.. 
        """
        self.geotype:GeometryType=GeometryType.POINT
        self.origin:np.ndarray=np.array(params.get("origin",[0]*3), dtype='float64')
        
        #frame treated as unit column vectors 
        if params.get("frame") is not None:
            self.frame:np.ndarray = np.array(params.get("frame"))
        else:
            u=np.array(params.get("u",[1,0,0]), dtype='float64')
            v=np.array(params.get("v",[0,1,0]), dtype='float64')
            w=np.array(params.get("w",[0,0,1]), dtype='float64')
            self.frame = np.vstack((u,v,w))
        self.r=params.get('r',0) #radius for circular geometries: circle, cylinder, sphere
        self.char_length=params.get("char_length", 1.0)
        self.limits = params.get('limits', []) #list of tuples describing extent of geometry relative to origin, number corresponds to geometry type. a cylinder will have one, a plane will have two, etc
        
        self._validate_frame()
        self.basis = self._get_basis()  # TODO: Implement when form error basis functions are ready
        #other params if necessary..
    
    def _validate_frame(self)->np.ndarray:
        
        return self.frame
    
    @property
    def u(self)->np.ndarray:
        return self.frame[0]
    
    @property
    def v(self)->np.ndarray:
        return self.frame[1]
    
    @property
    def w(self)->np.ndarray:
        return self.frame[2]
    
    def get_params(self):
        return dict(geotype=self.geotype, char_lenght=self.char_length, basis=self.basis)
    
    def is_null(self):
        return self.geotype.value<0
    
    @property
    def dim(self):
        return dimension(self.geotype)    

    def _get_basis(self):
        
        return basis_map.get(self.geotype, NullBasis)
    
    def convert(self, point:np.ndarray, forward:bool=True)->np.ndarray:
        """
        converts a point in space between coordinate systems
        """
        if forward:
            return self._convert_forward(point)
        else:
            return self._convert_backward(point)
        
    @abstractmethod
    def _convert_forward(self, xyz:np.ndarray)->np.ndarray:
        """
        convert a point from xyz to u(vw)
        e.g. for an axis, project onto tangent (first column of self.frame). v and w should be zeros (for now) 
        """
        pass
    
    @abstractmethod
    def _convert_backward(self, uvw:np.ndarray)->np.ndarray:
        """
         convert a point from uvw to xyz
        e.g. for an axis, just provides self.frame[:,0]*uvw[0]
         """
        pass
    
    @abstractmethod
    def coordinate(self, u:np.ndarray, v:np.ndarray|None=None, w:np.ndarray|None=None)->np.ndarray:
        """
        returns a point on this geometric object parameterized by u and optionally v,w
        """
        
    @abstractmethod
    def tangent(self, u:np.ndarray, v:np.ndarray|None=None, w:np.ndarray|None=None)->np.ndarray:
        """
        returns the tangent vector(s) at point u(vw)
        """
        pass
    
    @abstractmethod
    def normal(self, u:np.ndarray, v:np.ndarray|None=None, w:np.ndarray|None=None)->np.ndarray:
        """
        returns the normal vector(s) at point u(vw)
        """
        pass
    
    @abstractmethod
    def get_local_frame(u, v, w):
        """
        gets normal and tangent vectors at coordinate provided by uvw
        """
        pass

    def distance(self, other:'GeometryBase')->float:
        
        if other.geotype==GeometryType.POINT:
            return self._distance_point(other)
        elif dimension(self.geotype) <= dimension(other.geotype):
            return self._distance_other(other)
        else:
            return other._distance_other(self)
    
    @abstractmethod
    def _distance_point(self, point:'GeometryBase')->float:
        pass
    
    @abstractmethod
    def _distance_other(self, point:'GeometryBase')->float:
        """
        assumes dimension(self)<=dimension(other)
        """
        pass
    
    @abstractmethod
    def intersection(self, othergeometry:'GeometryBase')->'GeometryBase':
        """
        returns a valid intersected geometry (if exists), otherwise returns a null geometry
        """
        pass
    
    def derive(self, new_geo:int|GeometryType, **params)->'GeometryBase':
        """
        create a new geometry from this one. params contain target geo type, 
        """
        new_geo=GeometryType(new_geo)
        if new_geo==self.geotype:
            return self._derive_same(**params)
        else:
            # Pass new_geo to _derive_other
            params['new_geo'] = new_geo
            return self._derive_other(**params)
        
    @abstractmethod
    def _derive_same(self, **params)->'GeometryBase':
        """
        return a geometry of the same type that is translated, scaled, or with new radius (in case of circular geo's)
        example:
        
        #translation of origin in uvw frame
        du = params.get('du',np.ndarray([0]*3))
        dv = params.get('dv',np.ndarray([0]*3))
        dw = params.get('dw',np.ndarray([0]*3))
        
        #rotation of frame in uvw
        duw = params.get('duw',np.ndarray([0]*3))
        duv = params.get('duv',np.ndarray([0]*3))
        dvw = params.get('dvw',np.ndarray([0]*3))
        
        newr = params.get('r',self.r)
        
        new_origin = self.origin + np.ndarray([du, dv, dw])
        new_frame = rotate(self.frame, rotation_matrix(duw,duv,dvw)
        return type(self)(origin, frame, r=newr)
        """
        
    @abstractmethod
    def _derive_other(self, **params)->'GeometryBase':
        """
        derives a new geometry from this one, depending on parameters provided. if insufficient parameters are provided given geometry type, 
    return null geometry
        example:
        
        newgeo = GeometryType(params.get("geo_type", 0))
        if newgeo == GeometryType.POINT:
            pass
        elif newgeo == GeometryType.AXIS:
            tangent_at = params.get("tangent_at", self.origin)
            return axis that passes through and is parallel to tangent of tangent_at
            pass
        elif newgeo == GeometryType.PLANE:
            normal_at = params.get("normal_at", self.origin) 
            return plane that passes through and is normal to normal_at
        """

    @abstractmethod
    def derive_dual(self)->'GeometryBase':
        """
        get dual geometry
        """
        pass
    

    @staticmethod
    def rotate_frame(frame:np.ndarray, rx:float, ry:float, rz:float, deg = False):
        
        if deg:
            rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)

        rotations = []

        for i,r in enumerate([rx, ry, rz]):
            if np.isclose(r, 0):
                continue
            coshalfr = cos(r/2)
            sinhalfr = sin(r/2)
            scframe = sinhalfr*frame.copy()
            arg = [coshalfr] + list(scframe[i])
            
            rotation = transform.Rotation.from_quat(arg)
            rotations.append(rotation)
            
        rot_concat = transform.Rotation.concatenate(rotations)
        
        newframe = np.array(rot_concat.apply(frame))

        return newframe

    @abstractmethod
    def __contains__(self, other:'GeometryBase')->bool:
        """
        if dimension(self)<dimension(other):
            return True if self is contained in other, else False
        if dimension(self)==dimension(other):
            return True if self==other, else False
        if dimension(self)>dimension(other):
            return False
        """
    
    def __repr__(self):
        if self.geotype==GeometryType.NULL:
            return "NullGeometry()"
        
        typename = type(self).__name__
        outstrs = []
        oristr = ','.join([format(a,'0.3g') for a in self.origin])
        outstrs.append(f"origin=[{oristr}]")

        
        # coordstr = 'uvw'[:self.dim]
        coordstr='u'
        
        for coord,f in zip(coordstr,self.frame):
            if any(f):
                cstr = ','.join([format(a,'0.3g') for a in f])
                outstrs.append(f"{coord}=[{cstr}]")
        if self.r > 0:
            outstrs.append(f"r={self.r}")
        if self.limits:
            for (a,b),coord in zip(self.limits,'uvw'):
                outstrs.append(f"{coord}∈({a:0.3g},{b:0.3g})")
        
        outstr = ','.join(outstrs)
        
        return f"{typename}({outstr})"
        
class NullGeometry(GeometryBase):
    
    def __init__(self,**params):
        
        super().__init__(**params)
        
        self.geotype:GeometryType=GeometryType.NULL
        self.origin = np.array([0]*3)
        self.frame = np.array([[0]*3]*3)
        self.params = {}
        
    def _convert_forward(self, uvw:np.ndarray)->np.ndarray:
        return np.array([0]*3, dtype='float64')
    
    def _convert_backward(self, xyz:np.ndarray)->np.ndarray:
        return np.array([0]*3, dtype='float64')
    
    def coordinate(self, u:np.ndarray, v:np.ndarray|None=None, w:np.ndarray|None=None)->np.ndarray:
        return np.array([0]*3, dtype='float64')
        
    def tangent(self):
        return np.array([0]*3, dtype='float64')
    
    def normal(self, ):
        return np.array([0]*3, dtype='float64')
    
    def _distance_point(self, point:'GeometryBase')->float:
        return -1.0
    
    def _distance_other(self, point:'GeometryBase')->float:
        return -1.0
    
    def intersection(self, othergeometry:'GeometryBase')->'GeometryBase':
        return type(self)()
    
    def _derive_same(self, **params)->'GeometryBase':
        return type(self)()
    
    def _derive_other(self, **params)->'GeometryBase':
        return type(self)() 
           
    def derive_dual(self, **params)->'GeometryBase':
        return type(self)()

    def __contains__(self, other:'GeometryBase')->bool:
        return False
