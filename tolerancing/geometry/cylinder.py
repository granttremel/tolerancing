import numpy as np
from typing import TYPE_CHECKING
from .geometry import GeometryBase, GeometryType, NullGeometry
from .point import Point
from .axis import Axis

if TYPE_CHECKING:
    from .geometry import GeometryBase as GeometryBaseType
    from .plane import Plane


class Cylinder(GeometryBase):
    """
    Represents a 2D cylinder surface in 3D space.
    
    A cylinder is defined by:
    - origin: a point on the cylinder axis
    - frame: where u (first row) is the direction vector of the cylinder axis,
             v and w (second and third rows) are radial directions
    - r: radius of the cylinder
    """
    
    def __init__(self, **params):
        super().__init__(**params)
        self.geotype = GeometryType.CYLINDER
        
        # Ensure radius is positive
        if self.r <= 0:
            raise ValueError(f"Cylinder radius must be positive, got {self.r}")
        
        self._validate_frame()
    
    def _validate_frame(self):
        """
        Ensure frame forms an orthonormal basis.
        Similar to Axis but for a cylinder.
        """
        # Normalize u (the cylinder axis direction)
        umag = np.linalg.norm(self.frame[0])
        if umag < 1e-10:
            raise ValueError("Cylinder axis direction vector (u) is too small")
        
        self.frame[0] /= umag
        
        # Generate orthogonal v and w vectors (radial directions)
        u = self.frame[0]
        
        # Define world axes
        world_y = np.array([0, 1, 0])
        world_z = np.array([0, 0, 1])
        
        # Check if u is parallel to world-y
        dot_y = abs(np.dot(u, world_y))
        
        if dot_y > 0.99:  # u is nearly parallel to world-y
            # Use world-z as reference for v
            v = world_z - np.dot(world_z, u) * u
        else:
            # Use world-y as reference for v
            v = world_y - np.dot(world_y, u) * u
        
        # Normalize v
        vmag = np.linalg.norm(v)
        if vmag < 1e-10:
            raise ValueError("Cannot generate orthogonal basis - degenerate case")
        
        v = v / vmag
        
        # w is the cross product of u and v
        w = np.cross(u, v)
        
        # Assign back to frame
        self.frame[1] = v
        self.frame[2] = w
    
    def _convert_forward(self, xyz: np.ndarray) -> np.ndarray:
        """
        Convert a point from xyz to cylinder coordinates.
        u: distance along axis
        v, w: components in radial plane (can compute radius as sqrt(v²+w²))
        """
        # Translate to origin
        relative_point = xyz - self.origin
        
        # Project onto frame axes
        u_coord = np.dot(relative_point, self.u)  # Along axis
        v_coord = np.dot(relative_point, self.v)  # Radial component 1
        w_coord = np.dot(relative_point, self.w)  # Radial component 2
        
        return np.array([u_coord, v_coord, w_coord])
    
    def _convert_backward(self, uvw: np.ndarray) -> np.ndarray:
        """
        Convert a point from cylinder coordinates to xyz.
        """
        # Convert from frame coordinates to world coordinates
        xyz = self.origin + uvw[0] * self.u + uvw[1] * self.v + uvw[2] * self.w
        return xyz
    
    def coordinate(self, u: float = 0, v: float = None, w: float = None) -> np.ndarray:
        """
        Returns a point on the cylinder surface.
        u: position along axis
        v, w: if provided, used directly as radial components
        If only v is provided, it's treated as an angle (in radians)
        """
        if v is not None and w is None:
            # v is angle in radians
            angle = v
            v_coord = self.r * np.cos(angle)
            w_coord = self.r * np.sin(angle)
        elif v is not None and w is not None:
            # v and w are radial components - normalize to cylinder surface
            radial_mag = np.sqrt(v*v + w*w)
            if radial_mag < 1e-10:
                # Point on axis - use default radial direction
                v_coord = self.r
                w_coord = 0
            else:
                # Scale to cylinder surface
                v_coord = self.r * v / radial_mag
                w_coord = self.r * w / radial_mag
        else:
            # Default point on cylinder surface
            v_coord = self.r
            w_coord = 0
        
        return self.origin + u * self.u + v_coord * self.v + w_coord * self.w
    
    def tangent(self, u: float = 0, v: float = None, w: float = None) -> np.ndarray:
        """
        Returns a tangent vector at a point on the cylinder.
        The axial direction (u) is always tangent to the cylinder.
        """
        return self.u.copy()
    
    def normal(self, u: float = 0, v: float = None, w: float = None) -> np.ndarray:
        """
        Returns the outward normal vector at a point on the cylinder.
        The normal points radially outward from the axis.
        """
        if v is not None and w is None:
            # v is angle
            angle = v
            return np.cos(angle) * self.v + np.sin(angle) * self.w
        elif v is not None and w is not None:
            # Normalize radial direction
            radial_mag = np.sqrt(v*v + w*w)
            if radial_mag < 1e-10:
                return self.v.copy()  # Default radial direction
            return (v * self.v + w * self.w) / radial_mag
        else:
            # Default normal direction
            return self.v.copy()
    
    def _distance_point(self, point: 'GeometryBaseType') -> float:
        """
        Calculate distance from cylinder surface to a point.
        """
        # First get distance from point to cylinder axis
        axis_dist = self._distance_point_to_axis(point)
        
        # Distance from surface is |axis_dist - radius|
        return abs(axis_dist - self.r)
    
    def _distance_point_to_axis(self, point: 'GeometryBaseType') -> float:
        """
        Calculate distance from point to cylinder axis (not surface).
        """
        # Same calculation as axis-point distance
        P = point.origin
        P0 = self.origin
        d = self.u
        
        PP0 = P - P0
        projection = np.dot(PP0, d) * d
        perpendicular = PP0 - projection
        
        return np.linalg.norm(perpendicular)
    
    def _distance_other(self, other: 'GeometryBaseType') -> float:
        """
        Calculate distance to another geometry.
        """
        if other.geotype == GeometryType.CYLINDER:
            return self._distance_cylinder_cylinder(other)
        elif other.geotype == GeometryType.SPHERE:
            # Distance to sphere
            center_to_axis = self._distance_point_to_axis(Point(origin=other.origin))
            return max(0, abs(center_to_axis - self.r) - other.r)
        else:
            raise NotImplementedError(f"Distance from Cylinder to {other.geotype.name} not implemented")
    
    def _distance_axis(self, axis: 'GeometryBaseType') -> float:
        """
        Calculate distance from cylinder to axis.
        """
        # Check if parallel
        dot_product = np.dot(self.u, axis.u)
        if np.allclose(abs(dot_product), 1.0):
            # Parallel - find distance between axes
            axis_dist = self._distance_point_to_axis(Point(origin=axis.origin))
            return max(0, axis_dist - self.r)
        else:
            # Non-parallel case is complex
            raise NotImplementedError("Distance between non-parallel cylinder and axis not implemented")
    
    def _distance_plane(self, plane: 'GeometryBaseType') -> float:
        """
        Calculate distance from cylinder to plane.
        """
        # Check if cylinder axis is parallel to plane
        dot_product = np.dot(self.u, plane.u)
        
        if np.allclose(dot_product, 0):
            # Axis is parallel to plane
            plane_dist = abs(np.dot(plane.u, self.origin - plane.origin))
            return max(0, plane_dist - self.r)
        else:
            # Axis intersects plane - complex case
            # For now, just check if they intersect
            return 0.0
    
    def _distance_cylinder_cylinder(self, other: 'Cylinder') -> float:
        """
        Calculate distance between two cylinders.
        """
        # Check if parallel
        dot_product = np.dot(self.u, other.u)
        if np.allclose(abs(dot_product), 1.0):
            # Parallel cylinders
            axis_dist = self._distance_point_to_axis(Point(origin=other.origin))
            return max(0, axis_dist - self.r - other.r)
        else:
            raise NotImplementedError("Distance between non-parallel cylinders not implemented")
    
    def distance(self, other: 'GeometryBaseType') -> float:
        """
        Override distance to handle cylinder-specific cases.
        """
        if other.geotype == GeometryType.POINT:
            return self._distance_point(other)
        elif other.geotype == GeometryType.AXIS:
            return self._distance_axis(other)
        elif other.geotype == GeometryType.PLANE:
            return self._distance_plane(other)
        else:
            return super().distance(other)
    
    def intersection(self, other: 'GeometryBaseType') -> 'GeometryBaseType':
        """
        Calculate intersection with another geometry.
        """
        if other.geotype == GeometryType.POINT:
            return self._intersect_cylinder_point(other)
        elif other.geotype == GeometryType.AXIS:
            return self._intersect_cylinder_axis(other)
        elif other.geotype == GeometryType.PLANE:
            return self._intersect_cylinder_plane(other)
        elif other.geotype == GeometryType.CYLINDER:
            return self._intersect_cylinder_cylinder(other)
        else:
            return NullGeometry()
    
    def _intersect_cylinder_point(self, point: 'GeometryBaseType') -> 'GeometryBaseType':
        """
        Check if point lies on cylinder surface.
        """
        # Point is on cylinder if distance to axis equals radius
        axis_dist = self._distance_point_to_axis(point)
        if np.allclose(axis_dist, self.r):
            return Point(origin=point.origin.copy())
        return NullGeometry()
    
    def _intersect_cylinder_axis(self, axis: 'GeometryBaseType') -> 'GeometryBaseType':
        """
        Find intersection of cylinder with axis.
        """
        # Check if axis is parallel to cylinder axis
        dot_product = np.dot(self.u, axis.u)
        if np.allclose(abs(dot_product), 1.0):
            # Parallel - check if coaxial
            axis_dist = self._distance_point_to_axis(Point(origin=axis.origin))
            if np.allclose(axis_dist, 0):
                # Coaxial - return the axis
                return Axis(origin=axis.origin.copy(), frame=axis.frame.copy())
            return NullGeometry()
        
        # Non-parallel intersection is complex (can be 0, 1, or 2 points)
        # For now, return null
        return NullGeometry()
    
    def _intersect_cylinder_plane(self, plane: 'GeometryBaseType') -> 'GeometryBaseType':
        """
        Find intersection of cylinder with plane.
        """
        # Check if plane is perpendicular to cylinder axis
        dot_product = np.dot(plane.u, self.u)
        if np.allclose(abs(dot_product), 1.0):
            # Perpendicular - intersection is a circle
            # Find where cylinder axis intersects plane
            t = np.dot(plane.origin - self.origin, plane.u) / dot_product
            center = self.origin + t * self.u
            
            # Circle has same radius as cylinder
            # For now return null since we don't have Circle class
            return NullGeometry()
        
        # Non-perpendicular intersections are ellipses or more complex
        return NullGeometry()
    
    def _intersect_cylinder_cylinder(self, other: 'Cylinder') -> 'GeometryBaseType':
        """
        Find intersection of two cylinders.
        """
        # Only handle coaxial case
        dot_product = np.dot(self.u, other.u)
        if np.allclose(abs(dot_product), 1.0):
            # Check if coaxial
            axis_dist = self._distance_point_to_axis(Point(origin=other.origin))
            if np.allclose(axis_dist, 0):
                # Coaxial - return the axis
                return Axis(origin=self.origin.copy(), u=self.u.copy())
        return NullGeometry()
    
    def _derive_same(self, **params) -> 'GeometryBaseType':
        """
        Create a new cylinder with modified parameters.
        """
        # Translation in uvw frame
        du = params.get('du', 0)
        dv = params.get('dv', 0)
        dw = params.get('dw', 0)
        
        # New radius
        new_r = params.get('r', self.r)
        
        # new_origin = self.origin + du * self.u + dv * self.v + dw * self.w
        new_origin = du * self.u + dv * self.v + dw * self.w
        
        #rotate about origin in xyz coordinates
        rx = params.get('rx', 0)
        ry = params.get('ry', 0)
        rz = params.get('rz', 0)
        
        if not rx or not ry or not rz:
            new_frame = self.frame.copy()
        else:
            new_frame = self.rotate_frame(self.frame.copy(), rx, ry, rz)
        
        return Cylinder(origin=new_origin, frame=new_frame, r=new_r, reference=self)
    
    def _derive_other(self, **params) -> 'GeometryBaseType':
        """
        Derive a different type of geometry from this cylinder.
        """
        newgeo = GeometryType(params.get("geo_type", 0))
        
        if newgeo == GeometryType.AXIS:
            # Return the cylinder axis
            return Axis(origin=self.origin.copy(), frame=self.frame.copy())
        elif newgeo == GeometryType.POINT:
            # Point on cylinder surface
            u_pos = params.get('u', 0)
            angle = params.get('angle', 0)
            return Point(origin=self.coordinate(u_pos, angle),reference=self)
        else:
            return NullGeometry()
    
    def derive_dual(self, **params) -> 'GeometryBaseType':
        """
        For a cylinder, the dual could be its axis.
        """
        return Axis(origin=self.origin.copy(), u=self.u.copy())
    
    def __contains__(self, other: 'GeometryBaseType') -> bool:
        """
        Check if another geometry is contained in this cylinder.
        """
        if other.geotype == GeometryType.POINT:
            # Point is on cylinder surface if distance to axis equals radius
            axis_dist = self._distance_point_to_axis(other)
            return np.allclose(axis_dist, self.r)
        elif other.geotype == GeometryType.AXIS:
            # Axis is contained if it's the cylinder axis
            dot_product = np.dot(self.u, other.u)
            if np.allclose(abs(dot_product), 1.0):
                # Check if coaxial
                axis_dist = self._distance_point_to_axis(Point(origin=other.origin))
                return np.allclose(axis_dist, 0)
            return False
        elif other.geotype == GeometryType.CYLINDER:
            # Cylinders are equal if coaxial with same radius
            dot_product = np.dot(self.u, other.u)
            if np.allclose(abs(dot_product), 1.0) and np.allclose(self.r, other.r):
                axis_dist = self._distance_point_to_axis(Point(origin=other.origin))
                return np.allclose(axis_dist, 0)
            return False
        return False