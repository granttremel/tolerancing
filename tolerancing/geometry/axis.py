import numpy as np
from typing import TYPE_CHECKING
from .geometry import GeometryBase, GeometryType, NullGeometry
from .point import Point

if TYPE_CHECKING:
    from .geometry import GeometryBase


class Axis(GeometryBase):
    """
    Represents a 1D axis (line) in 3D space.
    
    An axis is defined by:
    - origin: a point on the axis
    - frame: where u (first column) is the direction vector of the axis
    """
    
    def __init__(self, **params):
        super().__init__(**params)
        self.geotype = GeometryType.AXIS

        self._validate_frame()

    def _validate_frame(self):
        # Normalize u (the axis direction)
        umag = np.linalg.norm(self.frame[0])
        if umag < 1e-10:
            raise ValueError("Axis direction vector (u) is too small")

        self.frame[0] /= umag

        # Generate orthogonal v and w vectors
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
        Convert a point from xyz to u(vw) coordinates.
        For an axis, project onto the tangent (u direction).
        v and w components are the perpendicular distances.
        """
        # Translate to origin
        relative_point = xyz - self.origin
        #treated as unit column vectors 
        # Project onto frame axes
        u_coord = np.dot(relative_point, self.u)
        v_coord = np.dot(relative_point, self.v)
        w_coord = np.dot(relative_point, self.w)
        
        return np.array([u_coord, v_coord, w_coord])
    
    def _convert_backward(self, uvw: np.ndarray) -> np.ndarray:
        """
        Convert a point from uvw to xyz coordinates.
        """
        # Convert from frame coordinates to world coordinates
        xyz = self.origin + uvw[0] * self.u + uvw[1] * self.v + uvw[2] * self.w
        return xyz
    
    def coordinate(self, u: float, v: float = None, w: float = None) -> np.ndarray:
        """
        Returns a point on the axis parameterized by u.
        v and w are ignored for an axis (they represent perpendicular distance).
        """
        return self.origin + u * self.u
    
    def get_local_frame(self, u, v, w):
        """
        an axis' frame is the same everywhere
        """
        return self.frame
    
    def tangent(self, u: float, v: float = None, w: float = None) -> np.ndarray:
        """
        Returns the tangent vector at any point on the axis.
        For a straight line, this is constant.
        """
        return self.u.copy()
    
    def normal(self, u: float, v: float = None, w: float = None) -> np.ndarray:
        """
        Returns the normal vector at a point on the axis.
        For an axis, there are infinitely many normals in the plane perpendicular to u.
        We return the v direction as a default normal.
        """
        return self.v.copy()
    
    def _distance_point(self, point: 'GeometryBase') -> float:
        """
        Calculate distance from this axis to a point.
        """
        # Distance from point to line: d = ||(P - P0) - ((P - P0)·d)d||
        P = point.origin
        P0 = self.origin
        d = self.u
        
        PP0 = P - P0
        projection = np.dot(PP0, d) * d
        perpendicular = PP0 - projection
        
        return np.linalg.norm(perpendicular)
    
    def _distance_other(self, other: 'GeometryBase') -> float:
        """
        Calculate distance to another geometry where dimension(self) <= dimension(other).
        Currently only handles axis-axis case.
        """
        if other.geotype == GeometryType.AXIS:
            return self._distance_axis_axis(other)
        else:
            raise NotImplementedError(f"Distance from Axis to {other.geotype.name} not implemented")
    
    def _distance_axis_axis(self, other: 'Axis') -> float:
        """
        Calculate minimum distance between two axes (lines in 3D).
        """
        # Check if axes are parallel
        dot_product = np.dot(self.u, other.u)
        if np.allclose(abs(dot_product), 1.0):
            # Parallel axes - distance is from any point on self to other
            return self._distance_point(Point(origin=other.origin))
        
        # For skew/intersecting axes
        n = np.cross(self.u, other.u)
        norm_n = np.linalg.norm(n)
        
        if norm_n < 1e-10:  # Parallel (shouldn't happen due to check above)
            return self._distance_point(Point(origin=other.origin))
        
        # Minimum distance between skew lines
        n_unit = n / norm_n
        distance = abs(np.dot(self.origin - other.origin, n_unit))
        return distance
    
    def intersection(self, other: 'GeometryBase') -> 'GeometryBase':
        """
        Calculate intersection with another geometry.
        """
        if other.geotype == GeometryType.POINT:
            return self._intersect_axis_point(other)
        elif other.geotype == GeometryType.AXIS:
            return self._intersect_axis_axis(other)
        else:
            return NullGeometry()
    
    def _intersect_axis_point(self, point: 'GeometryBase') -> 'GeometryBase':
        """
        Check if a point lies on this axis.
        """
        # Point is on axis if distance is ~0
        if np.allclose(self._distance_point(point), 0):
            return Point(origin=point.origin.copy())
        return NullGeometry()
    
    def _intersect_axis_axis(self, other: 'Axis') -> 'GeometryBase':
        """
        Find intersection point of two axes, if it exists.
        """
        # Check if axes are parallel
        dot_product = np.dot(self.u, other.u)
        if np.allclose(abs(dot_product), 1.0):
            # Parallel - check if coaxial
            if np.allclose(self._distance_point(Point(origin=other.origin)), 0):
                # Coaxial - return self
                return Axis(origin=self.origin.copy(), frame=self.frame.copy())
            return NullGeometry()
        
        # Find intersection point of two skew/intersecting lines
        # Using parametric form: P1 + t1*d1 = P2 + t2*d2
        P1, d1 = self.origin, self.u
        P2, d2 = other.origin, other.u
        
        P21 = P2 - P1
        d1_cross_d2 = np.cross(d1, d2)
        
        # Check if lines are skew (don't intersect)
        if not np.allclose(np.dot(P21, d1_cross_d2), 0):
            return NullGeometry()
        
        # Find intersection point
        norm_cross = np.dot(d1_cross_d2, d1_cross_d2)
        if norm_cross < 1e-10:
            return NullGeometry()
            
        t1 = np.dot(np.cross(P21, d2), d1_cross_d2) / norm_cross
        intersection = P1 + t1 * d1
        
        return Point(origin=intersection)
    
    def _derive_same(self, **params) -> 'GeometryBase':
        """
        Create a new axis with modified parameters.
        """
        # Translation in uvw frame
        du = params.get('du', 0)
        dv = params.get('dv', 0)
        dw = params.get('dw', 0)
        
        # Rotation of frame (not implemented for now)
        # TODO: Implement frame rotation when needed
        ru = params.get('ru',0)
        rv = params.get('rv',0)
        rw = params.get('rw',0)
        
        # new_origin = self.origin + du * self.u + dv * self.v + dw * self.w
        new_origin = du * self.u + dv * self.v + dw * self.w
        new_frame = self.frame.copy()
        
        return Axis(origin=new_origin, frame=new_frame, reference=self)
    
    def _derive_other(self, **params) -> 'GeometryBase':
        """
        Derive a different type of geometry from this axis.
        """
        newgeo = GeometryType(params.get("new_geo", params.get("geo_type", 0)))
        
        if newgeo == GeometryType.POINT:
            # Point along the axis
            du = params.get('du', 0)
            dv = params.get('dv', 0)
            dw = params.get('dw', 0)
            origin = self.origin + du * self.u + dv * self.v + dw * self.w
            from .point import Point
            return Point(origin=origin)
        elif newgeo == GeometryType.PLANE:
            # Plane perpendicular to axis
            du = params.get('du', 0)
            origin = self.origin + du * self.u
            from .plane import Plane
            return Plane(origin=origin, u=self.u)
        else:
            return NullGeometry()
    
    def derive_dual(self, **params):
        """
        Generate a plane perpendicular to this axis.
        """
        # Lazy import to avoid circular dependency
        from .plane import Plane
        # return Plane(origin=self.origin, u=self.u)
        return Plane(origin=[0,0,0], u=self.u, reference=self)
        
    def __contains__(self, other: 'GeometryBase') -> bool:
        """
        Check if another geometry is contained in this axis.
        """
        if other.geotype == GeometryType.POINT:
            # Point is on axis if distance is ~0
            return np.allclose(self._distance_point(other), 0)
        elif other.geotype == GeometryType.AXIS:
            # Axes are equal if they are coaxial
            dot_product = np.dot(self.u, other.u)
            if np.allclose(abs(dot_product), 1.0):
                # Check if origins align
                return np.allclose(self._distance_point(Point(origin=other.origin)), 0)
        return False
    
    def __repr__(self):
        """Simple representation of an axis."""
        return f"Axis(origin=[{self.origin[0]:.1f},{self.origin[1]:.1f},{self.origin[2]:.1f}],u=[{self.u[0]:.1f},{self.u[1]:.1f},{self.u[2]:.1f}])"