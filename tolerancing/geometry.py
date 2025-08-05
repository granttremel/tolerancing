

from enum import Enum
from typing import List, Dict, TYPE_CHECKING
import numpy as np

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

class Geometry:
    
    def __init__(self, geotype:GeometryType, origin=[0,0,0],frame=[0,0,1], r:float=0, form=None):
        
        self.geotype:GeometryType=geotype
        self.name:str=geotype.name.lower()
        #coordinates within part
        self.origin = np.array(origin)
        self.frame = np.array(frame) #parallel to axis, perpendicular to plane
        self.r=r
        self.form=form
        
        self.reference=None
        
        if geocalc.is_circular(geotype) and r==0:
            raise Exception(f"geometry of type {geotype.name}={geotype.value} must have radius!")

        if geocalc.has_form(geotype) and form is None:
            raise Exception(f"geometry of type {geotype.name}={geotype.value} must have form!")
        
    def set_origin(self, origin):
        self.origin=origin
        
    def set_frame(self, frame):
        self.frame=frame
    
    def set_reference(self, reference:'Geometry'):
        self.reference=reference

    def derive(self, offset=[0,0,0], orient=[0,0,0]):
        neworigin = self.origin + np.array(offset)
        newframe = self.frame + np.array(orient)
        newgeo = type(self)(self.geotype, neworigin, newframe, r=self.r, form=self.form)
        newgeo.set_reference(self.reference)
        return newgeo
        
    @property
    def aorigin(self):
        """Absolute origin"""
        if not self.reference:
            return self.origin
        else:
            return self.origin - self.reference.aorigin
    
    @property
    def aframe(self):
        """Absolute frame"""
        if not self.reference:
            return self.frame
        else:
            return self.frame - self.reference.aframe
    
    @classmethod
    def origin(cls):
        return cls(GeometryType(0),origin=[0,0,0])
    @classmethod
    def xaxis(cls):
        return cls(GeometryType(1),origin=[0,0,0],frame=[1,0,0])
    @classmethod
    def yaxis(cls):
        return cls(GeometryType(1),origin=[0,0,0],frame=[0,1,0])
    @classmethod
    def zaxis(cls):
        return cls(GeometryType(1),origin=[0,0,0],frame=[0,0,1])
    @classmethod
    def xyplane(cls):
        return cls(GeometryType(10),origin=[0,0,0],frame=[0,0,1])
    @classmethod
    def xzplane(cls):
        return cls(GeometryType(10),origin=[0,0,0],frame=[0,1,0])
    @classmethod
    def yzplane(cls):
        return cls(GeometryType(10),origin=[0,0,0],frame=[1,0,0])
    
    def __repr__(self):
        
        outstrs = []
        oristr = ','.join([format(o,"0.3f") for o in self.origin])
        framestr = ','.join([format(f,"0.3f") for f in self.frame])
        
        outstrs.append(f"origin=({oristr})")
        outstrs.append(f"frame=({framestr})")
        
        if self.r > 0:
            outstrs.append(f"r={self.r:0.3f}")
        if self.form:
            outstrs.append(f"form={self.form}")
        outstr=','.join(outstrs)
        
        return f"{self.geotype.name.capitalize()}({outstr})"

class GeoCalculator:
    
    _is_rectilinear = set([1,10,100])
    _is_circular = set([2,11,12,101])
    _has_form = set([9,99,999])
    
    _has_direction=set([1,10,11])
    
    def is_rectilinear(self, geo:GeometryType):
        return geo.value in self._is_rectilinear

    def is_circular(self, geo:GeometryType):
        return geo.value in self._is_circular

    def has_form(self, geo:GeometryType):
        return geo.value in self._has_form

    def _order(self, d1:'Geometry', d2:'Geometry'):
        
        return (d1,d2) if d1.geotype.value <= d2.geotype.value else (d2,d1)

    def contains(self, d1:'Geometry', d2:'Geometry')->bool:
        """Check if d2 contains d1 (d1 should be lower dimensional)"""
        d1,d2 = self._order(d1, d2)
        
        if d1.geotype == GeometryType.POINT:
            if d2.geotype == GeometryType.CYLINDER:
                return self._contains_cylinder(d1, d2)
            elif d2.geotype == GeometryType.SPHERE:
                return self._contains_sphere(d1, d2)
            elif d2.geotype == GeometryType.PLANE:
                # Point is on plane if distance is ~0
                return np.allclose(self._distance_point_plane(d1, d2), 0)
            elif d2.geotype == GeometryType.AXIS:
                # Point is on axis if distance is ~0
                return np.allclose(self._distance_point_axis(d1, d2), 0)
        
        raise ValueError(f"Contains check between {d1.geotype.name} and {d2.geotype.name} not supported")

    def _contains_cylinder(self, pt1:'Geometry', cyl2:'Geometry'):
        
        d = self._distance_point_cylinder(pt1, cyl2, mode="center")
        
        return d<=cyl2.r
    
    def _contains_sphere(self, pt1:'Geometry', sph2:'Geometry'):
        
        d = self._distance_point_sphere(pt1, sph2,mode="center")
        
        return d<=sph2.r

    def intersect(self, d1:'Geometry', d2:'Geometry')->'Geometry|None':
        
        #keep d1 smaller
        d1,d2 = self._order(d1, d2)
        
        # Check if intersection is defined
        key = (d1.geotype.value, d2.geotype.value)
        if key not in intersection_general:
            raise ValueError(f"Intersection between {d1.geotype.name} and {d2.geotype.name} not supported")
        
        # Point intersections
        if d1.geotype == GeometryType.POINT:
            if d2.geotype == GeometryType.POINT:
                return self._intersect_point_point(d1, d2)
            else:
                raise ValueError(f"Intersection between POINT and {d2.geotype.name} not supported")
        
        # Axis intersections
        elif d1.geotype == GeometryType.AXIS:
            if d2.geotype == GeometryType.AXIS:
                return self._intersect_axis_axis(d1, d2)
            elif d2.geotype == GeometryType.PLANE:
                return self._intersect_axis_plane(d1, d2)
            elif d2.geotype == GeometryType.CYLINDER:
                return self._intersect_axis_cylinder(d1, d2)
            elif d2.geotype == GeometryType.SPHERE:
                return self._intersect_axis_sphere(d1, d2)
        
        # Circle intersections
        elif d1.geotype == GeometryType.CIRCLE:
            if d2.geotype == GeometryType.PLANE:
                return self._intersect_circle_plane(d1, d2)
        
        # Plane intersections
        elif d1.geotype == GeometryType.PLANE:
            if d2.geotype == GeometryType.PLANE:
                return self._intersect_plane_plane(d1, d2)
            elif d2.geotype == GeometryType.CYLINDER:
                return self._intersect_plane_cylinder(d1, d2)
            elif d2.geotype == GeometryType.SPHERE:
                return self._intersect_plane_sphere(d1, d2)
        
        # Cylinder intersections
        elif d1.geotype == GeometryType.CYLINDER:
            if d2.geotype == GeometryType.CYLINDER:
                return self._intersect_cylinder_cylinder(d1, d2)
        
        # Sphere intersections
        elif d1.geotype == GeometryType.SPHERE:
            if d2.geotype == GeometryType.SPHERE:
                return self._intersect_sphere_sphere(d1, d2)
        
        else:
            raise ValueError(f"Intersection between {d1.geotype.name} and {d2.geotype.name} not implemented")

    def _intersect_point_point(self, pt1:'Geometry', pt2:'Geometry')->'Geometry|None':
        # Points only intersect if they are at the same location
        if np.allclose(pt1.aorigin, pt2.aorigin):
            return Geometry(GeometryType.POINT, pt1.aorigin)
        return None
    
    def _intersect_axis_axis(self, ax1:'Geometry', ax2:'Geometry')->'Geometry|None':
        # Check if axes are parallel
        if self.parallel(ax1,ax2):
            # Check if coaxial
            if np.allclose(ax1.aorigin, ax2.aorigin):
                return ax1  # Return one of the axes if coaxial
            return None
        
        # Find intersection point of two skew/intersecting lines
        # Using parametric form: P1 + t1*d1 = P2 + t2*d2
        P1, d1 = ax1.aorigin, ax1.aframe
        P2, d2 = ax2.aorigin, ax2.aframe
        
        # Solve for t1 and t2
        P21 = P2 - P1
        d1_cross_d2 = np.cross(d1, d2)
        
        # Check if lines are skew (don't intersect)
        if not np.allclose(np.dot(P21, d1_cross_d2), 0):
            return None
        
        # Find intersection point
        t1 = np.dot(np.cross(P21, d2), d1_cross_d2) / np.dot(d1_cross_d2, d1_cross_d2)
        intersection = P1 + t1 * d1
        
        return Geometry(GeometryType.POINT, intersection)
    
    def _intersect_axis_plane(self, ax:'Geometry', pl:'Geometry')->'Geometry|None':
        # Check if axis is parallel to plane
        if np.allclose(np.dot(ax.aframe, pl.aframe), 0):
            return None
        
        # Find intersection point
        # Line: P + t*d, Plane: n·(X - P0) = 0
        t = np.dot(pl.aorigin - ax.aorigin, pl.aframe) / np.dot(ax.aframe, pl.aframe)
        intersection = ax.aorigin + t * ax.aframe
        
        return Geometry(GeometryType.POINT, intersection)
    
    def _intersect_axis_cylinder(self, ax:'Geometry', cyl:'Geometry')->'Geometry|None':
        # Check if axis is parallel to cylinder axis
        if self.parallel(ax, cyl):
            # Check if coaxial
            dist = self._distance_axis_axis(ax, cyl)
            if np.allclose(dist, 0):
                return ax  # Return the axis if coaxial
        return None  # Non-parallel axis-cylinder intersection is complex
    
    def _intersect_axis_sphere(self, ax:'Geometry', sph:'Geometry')->'Geometry|None':
        # Find closest point on axis to sphere center
        # This is a quadratic problem
        # Line: P + t*d, Sphere: |X - C|² = r²
        P, d = ax.aorigin, ax.aframe
        C, r = sph.aorigin, sph.r
        
        PC = P - C
        a = np.dot(d, d)
        b = 2 * np.dot(PC, d)
        c = np.dot(PC, PC) - r*r
        
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return None
        elif np.allclose(discriminant, 0):
            t = -b / (2*a)
            return Geometry(GeometryType.POINT, P + t*d)
        else:
            # Two intersection points - return first one
            t1 = (-b - np.sqrt(discriminant)) / (2*a)
            return Geometry(GeometryType.POINT, P + t1*d)
    
    def _intersect_circle_plane(self, circ:'Geometry', pl:'Geometry')->'Geometry|None':
        # Complex - would need to check circle orientation vs plane
        raise NotImplementedError("Circle-plane intersection not yet implemented")
    
    def _intersect_plane_plane(self, pl1:'Geometry', pl2:'Geometry')->'Geometry|None':
        # Check if planes are parallel
        if self.parallel(pl1, pl2):
            return None
        
        # Intersection is an axis
        # Direction is cross product of normals
        direction = np.cross(pl1.aframe, pl2.aframe)
        direction = direction / np.linalg.norm(direction)
        
        # Find a point on the intersection line
        # Solve system of equations
        n1, n2 = pl1.aframe, pl2.aframe
        d1 = np.dot(n1, pl1.aorigin)
        d2 = np.dot(n2, pl2.aorigin)
        
        # Find the direction perpendicular to both line direction and n1
        perp = np.cross(direction, n1)
        
        # Solve for point
        t = (d2 - np.dot(n2, pl1.aorigin)) / np.dot(n2, perp)
        point = pl1.aorigin + t * perp
        
        return Geometry(GeometryType.AXIS, point, direction)
    
    def _intersect_plane_cylinder(self, pl:'Geometry', cyl:'Geometry')->'Geometry|None':
        # Check if plane is perpendicular to cylinder axis
        dot_product = np.dot(pl.aframe, cyl.aframe)
        if np.allclose(abs(dot_product), 1):
            # Perpendicular - intersection is a circle
            # Find intersection point of cylinder axis with plane
            t = np.dot(pl.aorigin - cyl.aorigin, pl.aframe) / dot_product
            center = cyl.aorigin + t * cyl.aframe
            return Geometry(GeometryType.CIRCLE, center, pl.aframe, r=cyl.r)
        # Non-perpendicular intersections are ellipses (not supported)
        raise ValueError("Plane-cylinder intersection resulting in ellipse not supported")
    
    def _intersect_plane_sphere(self, pl:'Geometry', sph:'Geometry')->'Geometry|None':
        # Find distance from sphere center to plane
        dist = abs(np.dot(sph.aorigin - pl.aorigin, pl.aframe))
        
        if dist > sph.r:
            return None
        elif np.allclose(dist, sph.r):
            # Tangent - single point
            point = sph.aorigin - dist * pl.aframe
            return Geometry(GeometryType.POINT, point)
        else:
            # Intersection is a circle
            # Find center of circle (projection of sphere center onto plane)
            center = sph.aorigin - np.dot(sph.aorigin - pl.aorigin, pl.aframe) * pl.aframe
            # Radius from Pythagoras
            radius = np.sqrt(sph.r**2 - dist**2)
            return Geometry(GeometryType.CIRCLE, center, pl.aframe, r=radius)
    
    def _intersect_cylinder_cylinder(self, cyl1:'Geometry', cyl2:'Geometry')->'Geometry|None':
        # Only handle coaxial case
        if self.parallel(cyl1, cyl2):
            dist = self._distance_axis_axis(cyl1, cyl2)
            if np.allclose(dist, 0):
                # Coaxial - return the axis
                return Geometry(GeometryType.AXIS, cyl1.aorigin, cyl1.aframe)
        return None
    
    def _intersect_sphere_sphere(self, sph1:'Geometry', sph2:'Geometry')->'Geometry|None':
        # Find distance between centers
        dist = np.linalg.norm(sph2.aorigin - sph1.aorigin)
        
        if dist > sph1.r + sph2.r or dist < abs(sph1.r - sph2.r):
            return None
        elif np.allclose(dist, sph1.r + sph2.r) or np.allclose(dist, abs(sph1.r - sph2.r)):
            # Tangent - single point
            direction = (sph2.aorigin - sph1.aorigin) / dist
            if dist == sph1.r + sph2.r:
                point = sph1.aorigin + sph1.r * direction
            else:
                point = sph1.aorigin + sph1.r * direction if sph1.r > sph2.r else sph1.aorigin - sph1.r * direction
            return Geometry(GeometryType.POINT, point)
        else:
            # Intersection is a circle
            # Find plane of intersection
            direction = (sph2.aorigin - sph1.aorigin) / dist
            # Distance from sph1 center to intersection plane
            a = (sph1.r**2 - sph2.r**2 + dist**2) / (2 * dist)
            center = sph1.aorigin + a * direction
            # Radius of intersection circle
            radius = np.sqrt(sph1.r**2 - a**2)
            return Geometry(GeometryType.CIRCLE, center, direction, r=radius)
            
        
    def parallel(self, d1:'Geometry', d2:'Geometry'):
        
        d1,d2 = self._order(d1, d2)
        
        if not d1.geotype.value in self._has_direction or not d2.geotype.value in self._has_direction:
            raise ValueError(f"Parallel check between {d1.geotype.name} and {d2.geotype.name} not supported") 
        
        # Check if directions are parallel (dot product is ±1)
        dot_product = np.dot(d1.aframe, d2.aframe)
        return np.allclose(abs(dot_product), 1)
        
    def distance(self, d1:'Geometry', d2:'Geometry', mode="min")->float:
        """
        Calculate distance between two datums.
        mode: "min" for minimum distance, "center" for center-to-center distance
        """
        d1,d2 = self._order(d1, d2)
        
        # Point distances
        if d1.geotype == GeometryType.POINT:
            if d2.geotype == GeometryType.POINT:
                return self._distance_point_point(d1, d2)
            elif d2.geotype == GeometryType.AXIS:
                return self._distance_point_axis(d1, d2)
            elif d2.geotype == GeometryType.PLANE:
                return self._distance_point_plane(d1, d2)
            elif d2.geotype == GeometryType.CYLINDER:
                return self._distance_point_cylinder(d1, d2, mode)
            elif d2.geotype == GeometryType.SPHERE:
                return self._distance_point_sphere(d1, d2, mode)
        
        # Axis distances
        elif d1.geotype == GeometryType.AXIS:
            if d2.geotype == GeometryType.AXIS:
                return self._distance_axis_axis(d1, d2)
            elif d2.geotype == GeometryType.PLANE:
                return self._distance_axis_plane(d1, d2)
            elif d2.geotype == GeometryType.CYLINDER:
                return self._distance_axis_cylinder(d1, d2, mode)
            elif d2.geotype == GeometryType.SPHERE:
                return self._distance_axis_sphere(d1, d2, mode)
        
        # Plane distances
        elif d1.geotype == GeometryType.PLANE:
            if d2.geotype == GeometryType.PLANE:
                return self._distance_plane_plane(d1, d2)
            elif d2.geotype == GeometryType.CYLINDER:
                return self._distance_plane_cylinder(d1, d2, mode)
            elif d2.geotype == GeometryType.SPHERE:
                return self._distance_plane_sphere(d1, d2, mode)
        
        # Cylinder distances
        elif d1.geotype == GeometryType.CYLINDER:
            if d2.geotype == GeometryType.CYLINDER:
                return self._distance_cylinder_cylinder(d1, d2, mode)
            elif d2.geotype == GeometryType.SPHERE:
                return self._distance_cylinder_sphere(d1, d2, mode)
        
        # Sphere distances
        elif d1.geotype == GeometryType.SPHERE:
            if d2.geotype == GeometryType.SPHERE:
                return self._distance_sphere_sphere(d1, d2, mode)
        
        else:
            raise ValueError(f"Distance between {d1.geotype.name} and {d2.geotype.name} not supported")
    
    def _distance_point_point(self, pt1:'Geometry', pt2:'Geometry')->float:
        
        delta = pt2.aorigin - pt1.aorigin
        return np.sqrt(np.dot(delta,delta))
    
    def _distance_point_axis(self, pt:'Geometry', ax:'Geometry')->float:
        # Distance from point to line
        # d = ||(P - P0) - ((P - P0)·d)d||
        P = pt.aorigin
        P0 = ax.aorigin
        d = ax.aframe
        
        PP0 = P - P0
        projection = np.dot(PP0, d) * d
        perpendicular = PP0 - projection
        
        return np.linalg.norm(perpendicular)
    
    def _distance_point_plane(self, pt:'Geometry', pl:'Geometry')->float:
        # Distance from point to plane
        # d = |n·(P - P0)|
        return abs(np.dot(pl.aframe, pt.aorigin - pl.aorigin))
    
    def _distance_point_cylinder(self, pt1:'Geometry', cyl2:'Geometry', mode="center")->float:
        
        ax = Geometry(GeometryType.AXIS, cyl2.aorigin, cyl2.aframe)
        d_axis = self._distance_point_axis(pt1, ax)
        
        if mode=="center":
            return d_axis
        elif mode=="min":
            return max(0, d_axis - cyl2.r)
        else:
            return -1
    
    def _distance_point_sphere(self, pt1:'Geometry', sph2:'Geometry', mode="center")->float:
        
        pt2 = Geometry(GeometryType.POINT, sph2.aorigin)
        d_sph = self._distance_point_point(pt1, pt2)
        
        if mode=="center":
            return d_sph
        elif mode=="min":
            return max(0, d_sph - sph2.r)
        else:
            return -1
        
    def _distance_axis_axis(self, ax1:'Geometry', ax2:'Geometry')->float:
        """Calculate minimum distance between two axes (lines in 3D)"""
        # For parallel axes
        if self.parallel(ax1, ax2):
            # Distance is the distance from any point on ax1 to ax2
            
            return self._distance_point_axis(Geometry(GeometryType.POINT, ax1.aorigin), ax2)
        
        # For skew/intersecting axes
        try:
            n = np.cross(ax1.aframe, ax2.aframe)
            normn = np.linalg.norm(n)
            if normn == 0:  # Parallel (shouldn't happen due to check above)
                
                return self._distance_point_axis(Geometry(GeometryType.POINT, ax1.aorigin), ax2)
            
            # Minimum distance between skew lines
            n_unit = n / normn
            distance = abs(np.dot(ax1.aorigin - ax2.aorigin, n_unit))
            return distance
        except Exception as e:
            raise ValueError(f"Error calculating distance between axes: {e}")
    
    def _distance_plane_plane(self, pl1:'Geometry', pl2:'Geometry')->float:
        # Planes only have distance if parallel
        if not self.parallel(pl1, pl2):
            return 0.0  # Intersecting planes have 0 distance
        
        # Distance between parallel planes
        # Project any point from pl2 onto pl1's normal
        return abs(np.dot(pl1.aframe, pl2.aorigin - pl1.aorigin))
    
    def _distance_axis_plane(self, ax:'Geometry', pl:'Geometry')->float:
        # If axis is not parallel to plane, they intersect (distance = 0)
        if not np.allclose(np.dot(ax.aframe, pl.aframe), 0):
            return 0.0
        
        # Axis is parallel to plane - find distance
        
        return self._distance_point_plane(Geometry(GeometryType.POINT, ax.aorigin), pl)
    
    def _distance_axis_cylinder(self, ax:'Geometry', cyl:'Geometry', mode="min")->float:
        # For non-parallel axes, this is complex
        # For now, only handle parallel case
        if self.parallel(ax, cyl):
            dist_between_axes = self._distance_axis_axis(ax, cyl)
            if mode == "min":
                return max(0, dist_between_axes - cyl.r)
            else:  # center
                return dist_between_axes
        else:
            raise NotImplementedError("Distance between non-parallel axis and cylinder not implemented")
    
    def _distance_axis_sphere(self, ax:'Geometry', sph:'Geometry', mode="min")->float:
        # Find closest point on axis to sphere center
        pt_on_axis = self._closest_point_on_axis(ax, sph.aorigin)
        dist_to_center = np.linalg.norm(pt_on_axis - sph.aorigin)
        
        if mode == "min":
            return max(0, dist_to_center - sph.r)
        else:  # center
            return dist_to_center
    
    def _distance_plane_cylinder(self, pl:'Geometry', cyl:'Geometry', mode="min")->float:
        # Check if cylinder axis is parallel to plane
        dot_product = np.dot(cyl.aframe, pl.aframe)
        
        if np.allclose(dot_product, 0):
            # Axis is parallel to plane
            dist_axis_to_plane = self._distance_axis_plane(cyl, pl)
            if mode == "min":
                return max(0, dist_axis_to_plane - cyl.r)
            else:  # center
                return dist_axis_to_plane
        else:
            # Axis intersects plane
            if mode == "min":
                return 0.0
            else:  # center distance to intersection point
                return abs(np.dot(pl.aorigin - cyl.aorigin, pl.aframe) / dot_product)
    
    def _distance_plane_sphere(self, pl:'Geometry', sph:'Geometry', mode="min")->float:
        # Distance from sphere center to plane
        dist_center = abs(np.dot(sph.aorigin - pl.aorigin, pl.aframe))
        
        if mode == "min":
            return max(0, dist_center - sph.r)
        else:  # center
            return dist_center
    
    def _distance_cylinder_cylinder(self, cyl1:'Geometry', cyl2:'Geometry', mode="min")->float:
        # Only handle parallel cylinders for now
        if self.parallel(cyl1, cyl2):
            dist_axes = self._distance_axis_axis(cyl1, cyl2)
            if mode == "min":
                return max(0, dist_axes - cyl1.r - cyl2.r)
            else:  # center
                return dist_axes
        else:
            raise NotImplementedError("Distance between non-parallel cylinders not implemented")
    
    def _distance_cylinder_sphere(self, cyl:'Geometry', sph:'Geometry', mode="min")->float:
        # Find closest point on cylinder axis to sphere center
        
        dist_axis_to_center = self._distance_point_axis(Geometry(GeometryType.POINT, sph.aorigin), cyl)
        
        if mode == "min":
            # Consider both radii
            return max(0, dist_axis_to_center - cyl.r - sph.r)
        else:  # center
            return dist_axis_to_center
    
    def _distance_sphere_sphere(self, sph1:'Geometry', sph2:'Geometry', mode="min")->float:
        # Distance between centers
        dist_centers = np.linalg.norm(sph2.aorigin - sph1.aorigin)
        
        if mode == "min":
            return max(0, dist_centers - sph1.r - sph2.r)
        else:  # center
            return dist_centers
    
    def _closest_point_on_axis(self, ax:'Geometry', point:np.ndarray)->np.ndarray:
        """Find the closest point on an axis to a given point"""
        P0 = ax.aorigin
        d = ax.aframe
        
        # Project point onto axis
        t = np.dot(point - P0, d)
        return P0 + t * d

geocalc = GeoCalculator()
