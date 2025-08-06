import numpy as np
from typing import TYPE_CHECKING
from .geometry import GeometryBase, GeometryType, NullGeometry

if TYPE_CHECKING:
    from .geometry import GeometryBase


class Point(GeometryBase):
    
    def __init__(self,**params):
        
        super().__init__(**params)
        # self.frame=np.array([[0]*3]*3, dtype='float64')
        self.geotype:GeometryType=GeometryType.POINT
        
    def _convert_forward(self, uvw:np.ndarray)->np.ndarray:
        return np.array([0]*3, dtype='float64')
    
    def _convert_backward(self, xyz:np.ndarray)->np.ndarray:
        return np.array([0]*3, dtype='float64')
    
    def coordinate(self, u:np.ndarray, v:np.ndarray|None=None, w:np.ndarray|None=None)->np.ndarray:
        return np.array([0]*3, dtype='float64')
        
    def tangent(self, u:np.ndarray, v:np.ndarray|None=None, w:np.ndarray|None=None)->np.ndarray:
        return np.array([0]*3, dtype='float64')
    
    def normal(self, u:np.ndarray, v:np.ndarray|None=None, w:np.ndarray|None=None)->np.ndarray:
        return np.array([0]*3, dtype='float64')
    
    def _distance_point(self, point:'GeometryBase')->float:
        delta = self.origin-point.origin
        return np.sqrt(np.dot(delta,delta.T)).item()
        
    def _distance_other(self, other:'GeometryBase')->float:
        #should already be handled..
        return other._distance_other(self)
    
    def intersection(self, othergeometry:'GeometryBase')->'GeometryBase':
        
        if self in othergeometry:
            return type(self)(origin=self.origin)
        
        return NullGeometry()
    
    def _derive_same(self, **params)->'GeometryBase':
        du = params.get('du',0)
        dv = params.get('dv',0)
        dw = params.get('dw',0)
        
        neworigin = self.origin + np.array([du,dv,dw])
        
        return type(self)(origin=neworigin)
    
    def _derive_other(self, **params)->'GeometryBase':
        return type(self)()
    
    def derive_dual(self)->'GeometryBase':
        #should be a space..?
        return type(self)(origin=self.origin)
    
    def __contains__(self, other:'GeometryBase')->bool:
        if other.geotype==GeometryType.POINT:
            if np.allclose(self.origin,other.origin):
                return True
        return False
    