import numpy as np
from typing import TYPE_CHECKING
from .geometry import GeometryBase, GeometryType, NullGeometry
from .point import Point
from .axis import Axis

if TYPE_CHECKING:
    from .geometry import GeometryBase as GeometryBaseType
    # from .axis import Axis


class Plane(GeometryBase):
    """
    Represents a 2D plane in 3D space.
    
    A plane is defined by:
    - origin: a point on the plane
    - frame: where u (first row) is the normal vector to the plane,
             v and w (second and third rows) are tangent vectors in the plane
    """
    
    def __init__(self, **params):
        super().__init__(**params)
        self.geotype = GeometryType.PLANE
        self._validate_frame()
    
    def _validate_frame(self):
        """
        Ensure frame forms an orthonormal basis with u as the normal.
        """
        # Normalize u (the normal vector)
        umag = np.linalg.norm(self.frame[0])
        if umag < 1e-10:
            raise ValueError("Plane normal vector (u) is too small")
        
        self.frame[0] /= umag
        
        # Generate orthogonal v and w vectors in the plane
        u = self.frame[0]
        
        # Define world axes
        world_x = np.array([1, 0, 0])
        world_y = np.array([0, 1, 0])
        world_z = np.array([0, 0, 1])
        
        # Find the world axis least aligned with u to use as reference
        dots = [abs(np.dot(u, world_x)), abs(np.dot(u, world_y)), abs(np.dot(u, world_z))]
        min_idx = np.argmin(dots)
        
        if min_idx == 0:
            ref_vec = world_x
        elif min_idx == 1:
            ref_vec = world_y
        else:
            ref_vec = world_z
        
        # Project reference vector onto plane
        v = ref_vec - np.dot(ref_vec, u) * u
        vmag = np.linalg.norm(v)
        
        if vmag < 1e-10:
            raise ValueError("Cannot generate orthogonal basis - degenerate case")
        
        v = v / vmag
        
        # w completes the right-handed orthonormal basis
        w = np.cross(u, v)
        
        # Assign back to frame
        self.frame[1] = v
        self.frame[2] = w
    
    def _convert_forward(self, xyz: np.ndarray) -> np.ndarray:
        """
        Convert a point from xyz to uvw coordinates.
        For a plane, u is the signed distance from the plane,
        v and w are coordinates within the plane.
        """
        # Translate to origin
        relative_point = xyz - self.origin
        
        # Project onto frame axes
        u_coord = np.dot(relative_point, self.u)  # Distance from plane
        v_coord = np.dot(relative_point, self.v)  # Coordinate in plane
        w_coord = np.dot(relative_point, self.w)  # Coordinate in plane
        
        return np.array([u_coord, v_coord, w_coord])
    
    def _convert_backward(self, uvw: np.ndarray) -> np.ndarray:
        """
        Convert a point from uvw to xyz coordinates.
        """
        # Convert from frame coordinates to world coordinates
        xyz = self.origin + uvw[0] * self.u + uvw[1] * self.v + uvw[2] * self.w
        return xyz
    
    def coordinate(self, u: float = 0, v: float = 0, w: float = 0) -> np.ndarray:
        """
        Returns a point on the plane parameterized by v and w.
        u represents distance from the plane (typically 0 for points on the plane).
        """
        return self.origin + u * self.u + v * self.v + w * self.w
    
    def get_local_frame(self, u, v, w):
        """
        a plane's frame is the same everywhere
        """
        return self.frame
    
    def tangent(self, u: float = 0, v: float = 0, w: float = 0) -> np.ndarray:
        """
        Returns a tangent vector in the plane.
        For a plane, we return the v direction as a default tangent.
        """
        return self.v.copy()
    
    def normal(self, u: float = 0, v: float = 0, w: float = 0) -> np.ndarray:
        """
        Returns the normal vector to the plane.
        For a plane, this is constant (the u direction).
        """
        return self.u.copy()
    
    def _distance_point(self, point: 'GeometryBaseType') -> float:
        """
        Calculate distance from this plane to a point.
        Distance from point to plane: d = |n·(P - P0)|
        """
        return abs(np.dot(self.u, point.origin - self.origin))
    
    def _distance_other(self, other: 'GeometryBaseType') -> float:
        """
        Calculate distance to another geometry where dimension(self) <= dimension(other).
        Since plane is 2D, this handles higher dimensional geometries.
        """
        if other.geotype == GeometryType.SPHERE:
            # Distance to sphere center minus radius
            center_dist = self._distance_point(Point(origin=other.origin))
            return max(0, center_dist - other.r)
        elif other.geotype == GeometryType.VOLUME:
            # Would need to implement volume distance
            raise NotImplementedError(f"Distance from Plane to {other.geotype.name} not implemented")
        else:
            raise NotImplementedError(f"Distance from Plane to {other.geotype.name} not implemented")
    
    def _distance_axis(self, axis: 'GeometryBaseType') -> float:
        """
        Calculate distance from plane to axis.
        """
        # If axis is not parallel to plane, they intersect (distance = 0)
        if not np.allclose(np.dot(axis.u, self.u), 0):
            return 0.0
        
        # Axis is parallel to plane - find distance
        return self._distance_point(Point(origin=axis.origin))
    
    def _distance_plane(self, other: 'Plane') -> float:
        """
        Calculate distance between two planes.
        """
        # Check if planes are parallel
        dot_product = np.dot(self.u, other.u)
        if not np.allclose(abs(dot_product), 1.0):
            return 0.0  # Intersecting planes have 0 distance
        
        # Distance between parallel planes
        return abs(np.dot(self.u, other.origin - self.origin))
    
    def distance(self, other: 'GeometryBaseType') -> float:
        """
        Override distance to handle plane-specific cases.
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
            return self._intersect_plane_point(other)
        elif other.geotype == GeometryType.AXIS:
            return self._intersect_plane_axis(other)
        elif other.geotype == GeometryType.PLANE:
            return self._intersect_plane_plane(other)
        else:
            return NullGeometry()
    
    def _intersect_plane_point(self, point: 'GeometryBaseType') -> 'GeometryBaseType':
        """
        Check if a point lies on this plane.
        """
        # Point is on plane if distance is ~0
        if np.allclose(self._distance_point(point), 0):
            return Point(origin=point.origin.copy())
        return NullGeometry()
    
    def _intersect_plane_axis(self, axis: 'GeometryBaseType') -> 'GeometryBaseType':
        """
        Find intersection of plane with axis.
        """
        # Check if axis is parallel to plane
        dot_product = np.dot(axis.u, self.u)
        if np.allclose(dot_product, 0):
            # Parallel - check if axis lies in plane
            if np.allclose(self._distance_point(Point(origin=axis.origin)), 0):
                # Import here to avoid circular dependency
                from .axis import Axis
                return Axis(origin=axis.origin.copy(), frame=axis.frame.copy())
            return NullGeometry()
        
        # Find intersection point
        # Line: P + t*d, Plane: n·(X - P0) = 0
        t = np.dot(self.origin - axis.origin, self.u) / dot_product
        intersection = axis.origin + t * axis.u
        
        return Point(origin=intersection)
    
    def _intersect_plane_plane(self, other: 'Plane') -> 'GeometryBaseType':
        """
        Find intersection of two planes (returns an axis).
        """
        # Check if planes are parallel
        dot_product = np.dot(self.u, other.u)
        if np.allclose(abs(dot_product), 1.0):
            return NullGeometry()
        
        # Intersection is an axis
        # Direction is cross product of normals
        direction = np.cross(self.u, other.u)
        direction = direction / np.linalg.norm(direction)
        
        # Find a point on the intersection line
        # We need to solve for a point that lies on both planes
        n1, n2 = self.u, other.u
        d1 = np.dot(n1, self.origin)
        d2 = np.dot(n2, other.origin)
        
        # Find the direction perpendicular to both line direction and n1
        perp = np.cross(direction, n1)
        perp_norm = np.linalg.norm(perp)
        
        if perp_norm < 1e-10:
            # Use a different approach if perp is too small
            perp = np.cross(direction, n2)
            perp = perp / np.linalg.norm(perp)
            t = (d1 - np.dot(n1, other.origin)) / np.dot(n1, perp)
            point = other.origin + t * perp
        else:
            perp = perp / perp_norm
            t = (d2 - np.dot(n2, self.origin)) / np.dot(n2, perp)
            point = self.origin + t * perp
        
        # Import here to avoid circular dependency
        from .axis import Axis
        return Axis(origin=point, u=direction)
    
    def _derive_same(self, **params) -> 'GeometryBaseType':
        """
        Create a new plane with modified parameters.
        """
        # Translation in uvw frame
        du = params.get('du', 0)  # Normal direction
        dv = params.get('dv', 0)  # In-plane
        dw = params.get('dw', 0)  # In-plane
        
        # Rotation of frame (not implemented for now)
        # TODO: Implement frame rotation when needed
        
        new_origin = self.origin + du * self.u + dv * self.v + dw * self.w
        new_frame = self.frame.copy()
        
        return Plane(origin=new_origin, frame=new_frame)
    
    def _derive_other(self, **params) -> 'GeometryBaseType':
        """
        Derive a different type of geometry from this plane.
        """
        newgeo = GeometryType(params.get("new_geo", params.get("geo_type", 0)))
        
        if newgeo == GeometryType.POINT:
            # Point at specified coordinates on plane
            v_coord = params.get('v', 0)
            w_coord = params.get('w', 0)
            return Point(origin=self.coordinate(0, v_coord, w_coord))
        elif newgeo == GeometryType.AXIS:
            # Axis in plane at specified location and direction
            at_point = params.get('at_point', self.origin)
            direction = params.get('direction', self.v)
            # Project direction onto plane
            direction = direction - np.dot(direction, self.u) * self.u
            if np.linalg.norm(direction) < 1e-10:
                return NullGeometry()
            from .axis import Axis
            return Axis(origin=at_point, u=direction)
        elif newgeo == GeometryType.CYLINDER:
            # Cylinder perpendicular to plane
            du = params.get('du', 0)
            dv = params.get('dv', 0)
            dw = params.get('dw', 0)
            origin = self.origin + du * self.u + dv * self.v + dw * self.w
            direction = params.get('direction', self.u)  # Default to plane normal
            radius = params.get('r', 1.0)
            from .cylinder import Cylinder
            return Cylinder(origin=origin, u=direction, r=radius)
        else:
            return NullGeometry()
        
    def derive_dual(self, **params):
        return Axis(origin=self.origin, u=self.u)
    
    def __contains__(self, other: 'GeometryBaseType') -> bool:
        """
        Check if another geometry is contained in this plane.
        """
        if other.geotype == GeometryType.POINT:
            # Point is in plane if distance is ~0
            return np.allclose(self._distance_point(other), 0)
        elif other.geotype == GeometryType.AXIS:
            # Axis is in plane if it's parallel and origin is in plane
            if np.allclose(np.dot(other.u, self.u), 0):
                return np.allclose(self._distance_point(Point(origin=other.origin)), 0)
            return False
        elif other.geotype == GeometryType.PLANE:
            # Planes are equal if they have same normal and contain each other's origins
            dot_product = np.dot(self.u, other.u)
            if np.allclose(abs(dot_product), 1.0):
                return np.allclose(self._distance_point(Point(origin=other.origin)), 0)
            return False
        return False
    
    def __repr__(self):
        """Simple representation of a plane."""
        return f"Plane(origin=[{self.origin[0]:.1f},{self.origin[1]:.1f},{self.origin[2]:.1f}],u=[{self.u[0]:.1f},{self.u[1]:.1f},{self.u[2]:.1f}])"