
from typing import List, Dict, TYPE_CHECKING
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

from .geometry import Geometry, GeometryType, GeoCalculator

if TYPE_CHECKING:
    from .component import Component

geocalc=GeoCalculator()

class Datum:
    
    def __init__(self, geo: Geometry = None, dimension: 'Dimension' = None, name: str = None):
        """
        Create a datum either from existing geometry or from a dimension relationship.
        
        Args:
            geo: Existing geometry (for primary datums)
            dimension: Dimension relationship (for derived datums) 
            name: Optional name for the datum
        """
        self.name = name
        self.dimension = dimension
        self.tol: Tolerance = None
        self.fixed = False
        self.reference = None
        self.parent = None
        
        if geo is not None:
            # Primary datum with explicit geometry
            self.geo = geo
        elif dimension is not None:
            # Derived datum - calculate geometry from dimension
            if not dimension.is_geometry_valid(dimension.constraints.get('target_geometry')):
                raise ValueError(f"Dimension {dimension.dimension_type} insufficient for target geometry")
            target_geo = dimension.constraints.get('target_geometry')
            self.geo = dimension.calculate_geometry(target_geo)
        else:
            raise ValueError("Must provide either geometry or dimension")
    
    @classmethod
    def from_geometry(cls, geometry_type: GeometryType, origin=[0,0,0], frame=[0,0,1], 
                     r: float = 0, name: str = None) -> 'Datum':
        """Create a primary datum from geometry parameters"""
        geo = Geometry(geometry_type, origin, frame, r)
        return cls(geo=geo, name=name)
    
    @classmethod 
    def from_dimension(cls, reference_datums: List['Datum'], dimension_type: str,
                      values: List[float], target_geometry: GeometryType,
                      constraints: Dict = None, name: str = None) -> 'Datum':
        """Create a derived datum from dimensional relationship"""
        constraints = constraints or {}
        constraints['target_geometry'] = target_geometry
        
        dimension = Dimension(reference_datums, dimension_type, values, constraints)
        return cls(dimension=dimension, name=name)
    
    def set_reference(self, reference: 'Datum'):
        self.reference = reference
        self.geo.set_reference(reference.geo)
    
    def set_parent(self, component: 'Component'):
        self.parent = component
        
    def set_tolerance(self, tolerance: 'Tolerance'):
        self.tol = tolerance
        
    def is_defined(self) -> bool:
        """Check if datum is fully geometrically defined"""
        if self.dimension:
            target_geo = self.dimension.constraints.get('target_geometry')
            return self.dimension.is_geometry_valid(target_geo)
        return True  # Primary datums are always defined
    
    def get_construction_chain(self) -> List['Datum']:
        """Get the chain of datums this datum depends on"""
        if not self.dimension:
            return [self]  # Primary datum
        
        chain = []
        for ref_datum in self.dimension.reference_datums:
            chain.extend(ref_datum.get_construction_chain())
        chain.append(self)
        return chain
    
    # Convenience methods for common datum creation patterns
    def offset_distance(self, distance: float, direction: np.ndarray, 
                       target_geometry: GeometryType, name: str = None) -> 'Datum':
        """Create datum offset by distance in specified direction"""
        return Datum.from_dimension(
            reference_datums=[self],
            dimension_type='offset_distance', 
            values=[distance],
            target_geometry=target_geometry,
            constraints={'direction': direction},
            name=name
        )
    
    def offset_xy(self, dx: float, dy: float, target_geometry: GeometryType,
                 direction: np.ndarray = None, name: str = None) -> 'Datum':
        """Create datum offset by X,Y distances"""
        constraints = {}
        if direction is not None:
            constraints['direction'] = direction
        
        return Datum.from_dimension(
            reference_datums=[self],
            dimension_type='offset_xy',
            values=[dx, dy], 
            target_geometry=target_geometry,
            constraints=constraints,
            name=name
        )
    
    def offset_xyz(self, dx: float, dy: float, dz: float, name: str = None) -> 'Datum':
        """Create point offset by X,Y,Z distances"""
        return Datum.from_dimension(
            reference_datums=[self],
            dimension_type='offset_xyz',
            values=[dx, dy, dz],
            target_geometry=GeometryType.POINT,
            name=name
        )
    
    def angle_from(self, angle: float, reference_direction: np.ndarray,
                  target_geometry: GeometryType, name: str = None) -> 'Datum':
        """Create datum at angle from this datum"""
        return Datum.from_dimension(
            reference_datums=[self],
            dimension_type='angle_from_axis',
            values=[angle],
            target_geometry=target_geometry,
            constraints={'reference_direction': reference_direction},
            name=name
        )
    
    #this should be a construction not a datum..
    @staticmethod
    def intersect_planes(plane1: 'Datum', plane2: 'Datum', name: str = None) -> 'Datum':
        """Create axis from intersection of two planes"""
        return Datum.from_dimension(
            reference_datums=[plane1, plane2],
            dimension_type='intersect_two_planes',
            values=[],
            target_geometry=GeometryType.AXIS,
            name=name
        )
    
    def tangent_plane(self, direction: np.ndarray, name: str = None) -> 'Datum':
        """Create plane tangent to this datum (must be cylinder)"""
        if self.geo.geotype != GeometryType.CYLINDER:
            raise ValueError("Tangent plane only supported for cylinders")
        
        return Datum.from_dimension(
            reference_datums=[self],
            dimension_type='tangent_to_cylinder',
            values=[],
            target_geometry=GeometryType.PLANE,
            constraints={'direction': direction},
            name=name
        )
    
    def __repr__(self):
        geo_str = f"{self.geo.geotype.name}" if self.geo else "undefined"
        name_str = f"'{self.name}'" if self.name else "unnamed"
        return f"Datum({name_str}, {geo_str})"


class Dimension:
    """
    Represents a dimensional relationship between datums.
    Captures the toleranceable quantity (distance, angle, etc.) and geometric constraints.
    """
    
    _dimension_counter = 0  # Class variable to track dimension count
    
    def __init__(self, reference_datums: List['Datum'], dimension_type: str, 
                 values: List[float], constraints: Dict = None, name: str = None):
        """
        reference_datums: List of datums this dimension references
        dimension_type: Type of dimension ('offset_distance', 'offset_xy', 'angle', etc.)
        values: The dimensional values (distances, angles, coordinates)
        constraints: Additional geometric constraints (direction vectors, etc.)
        name: Optional name for the dimension
        """
        # Increment counter and assign ID
        Dimension._dimension_counter += 1
        self.id = f"DIM_{Dimension._dimension_counter:04d}"
        
        # Generate name if not provided
        if name:
            self.name = name
        elif reference_datums and len(reference_datums) > 0:
            ref_name = reference_datums[0].name or f"datum_{id(reference_datums[0])}"
            self.name = f"{ref_name}_{dimension_type}_{self.id}"
        else:
            self.name = f"{dimension_type}_{self.id}"
        
        self.reference_datums = reference_datums
        self.dimension_type = dimension_type
        self.values = np.array(values) if isinstance(values, list) else values
        self.constraints = constraints or {}
        self.tolerance = None
    
    def set_tolerance(self, tolerance: 'Tolerance'):
        self.tolerance = tolerance
    
    def is_geometry_valid(self, target_geometry: GeometryType) -> bool:
        """Check if this dimension provides enough constraints for target geometry"""
        return DIMENSION_VALIDATORS[self.dimension_type](self, target_geometry)
    
    def calculate_geometry(self, target_geometry: GeometryType) -> 'Geometry':
        """Calculate the actual geometry from this dimension"""
        return DIMENSION_CALCULATORS[self.dimension_type](self, target_geometry)


# Dimension type validators - check if dimension provides enough constraints
DIMENSION_VALIDATORS = {
    'offset_distance': lambda dim, geo: (
        len(dim.reference_datums) == 1 and 
        'direction' in dim.constraints and
        geo in [GeometryType.POINT, GeometryType.PLANE, GeometryType.SPHERE, GeometryType.AXIS, GeometryType.CYLINDER]
    ),
    'offset_xy': lambda dim, geo: (
        len(dim.reference_datums) == 1 and
        len(dim.values) == 2 and
        geo in [GeometryType.POINT, GeometryType.AXIS]
    ),
    'offset_xyz': lambda dim, geo: (
        len(dim.reference_datums) == 1 and
        len(dim.values) == 3 and
        geo == GeometryType.POINT
    ),
    'angle_from_axis': lambda dim, geo: (
        len(dim.reference_datums) == 1 and
        dim.reference_datums[0].geo.geotype in [GeometryType.AXIS, GeometryType.PLANE] and
        'reference_direction' in dim.constraints and
        geo in [GeometryType.AXIS, GeometryType.PLANE]
    ),
    'intersect_two_planes': lambda dim, geo: (
        len(dim.reference_datums) == 2 and
        all(d.geo.geotype == GeometryType.PLANE for d in dim.reference_datums) and
        geo == GeometryType.AXIS
    ),
    'tangent_to_cylinder': lambda dim, geo: (
        len(dim.reference_datums) == 1 and
        dim.reference_datums[0].geo.geotype == GeometryType.CYLINDER and
        'direction' in dim.constraints and
        geo == GeometryType.PLANE
    )
}

# Dimension calculator functions
def _calculate_offset_distance(dimension: 'Dimension', target_geometry: GeometryType) -> 'Geometry':
    """Calculate geometry offset by distance in specified direction"""
    ref_datum = dimension.reference_datums[0]
    distance = dimension.values if np.isscalar(dimension.values) else dimension.values[0]
    direction = np.array(dimension.constraints['direction'])
    direction = direction / np.linalg.norm(direction)  # normalize
    
    new_origin = ref_datum.geo.aorigin + distance * direction
    
    if target_geometry == GeometryType.POINT:
        return Geometry(target_geometry, new_origin)
    elif target_geometry == GeometryType.PLANE:
        return Geometry(target_geometry, new_origin, direction)
    elif target_geometry == GeometryType.AXIS:
        # Keep the same orientation as reference datum
        return Geometry(target_geometry, new_origin, ref_datum.geo.aframe)
    elif target_geometry == GeometryType.CYLINDER:
        # For cylinder, use direction as axis direction and get radius from constraints
        radius = dimension.constraints.get('radius', 0)
        return Geometry(target_geometry, new_origin, direction, r=radius)
    elif target_geometry == GeometryType.SPHERE:
        radius = dimension.constraints.get('radius', 0)
        return Geometry(target_geometry, new_origin, r=radius)

def _calculate_offset_xy(dimension: 'Dimension', target_geometry: GeometryType) -> 'Geometry':
    """Calculate geometry offset by X,Y distances from reference"""
    ref_datum = dimension.reference_datums[0]
    dx, dy = dimension.values[0], dimension.values[1]
    
    # Assume reference provides coordinate system (e.g., a plane)
    if ref_datum.geo.geotype == GeometryType.PLANE:
        # Use plane's coordinate system
        x_axis = np.array([1, 0, 0])  # Would need proper coordinate system
        y_axis = np.array([0, 1, 0])
        new_origin = ref_datum.geo.aorigin + dx * x_axis + dy * y_axis
    else:
        # Fallback to global coordinates
        new_origin = ref_datum.geo.aorigin + np.array([dx, dy, 0])
    
    if target_geometry == GeometryType.POINT:
        return Geometry(target_geometry, new_origin)
    elif target_geometry == GeometryType.AXIS:
        direction = dimension.constraints.get('direction', [0, 0, 1])
        return Geometry(target_geometry, new_origin, direction)

def _calculate_offset_xyz(dimension: 'Dimension', target_geometry: GeometryType) -> 'Geometry':
    """Calculate point offset by X,Y,Z distances"""
    ref_datum = dimension.reference_datums[0]
    dx, dy, dz = dimension.values[0], dimension.values[1], dimension.values[2]
    new_origin = ref_datum.geo.aorigin + np.array([dx, dy, dz])
    return Geometry(target_geometry, new_origin)

def _calculate_angle_from_axis(dimension: 'Dimension', target_geometry: GeometryType) -> 'Geometry':
    """Calculate geometry at angle from reference axis/plane"""
    ref_datum = dimension.reference_datums[0]
    angle = dimension.values if np.isscalar(dimension.values) else dimension.values[0]
    ref_direction = np.array(dimension.constraints['reference_direction'])
    
    # Rotate reference direction by angle around reference axis
    if ref_datum.geo.geotype == GeometryType.AXIS:
        axis = ref_datum.geo.aframe
    else:  # PLANE
        axis = ref_datum.geo.aframe
    
    # Simple rotation (would need proper rotation matrix for full implementation)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # This is simplified - would need proper 3D rotation
    new_direction = ref_direction  # Placeholder
    
    if target_geometry == GeometryType.AXIS:
        return Geometry(target_geometry, ref_datum.geo.aorigin, new_direction)
    elif target_geometry == GeometryType.PLANE:
        return Geometry(target_geometry, ref_datum.geo.aorigin, new_direction)

def _calculate_intersect_two_planes(dimension: 'Dimension', target_geometry: GeometryType) -> 'Geometry':
    """Calculate axis from intersection of two planes"""
    plane1 = dimension.reference_datums[0].geo
    plane2 = dimension.reference_datums[1].geo
    
    # Use existing intersection calculation
    return geocalc.intersect(plane1, plane2)

def _calculate_tangent_to_cylinder(dimension: 'Dimension', target_geometry: GeometryType) -> 'Geometry':
    """Calculate plane tangent to cylinder"""
    cyl_datum = dimension.reference_datums[0]
    direction = np.array(dimension.constraints['direction'])
    
    # Find tangent point on cylinder surface
    # This would need proper tangent calculation
    tangent_point = cyl_datum.geo.aorigin + cyl_datum.geo.r * direction
    return Geometry(target_geometry, tangent_point, direction)

# Dimension calculators - compute actual geometry
DIMENSION_CALCULATORS = {
    'offset_distance': _calculate_offset_distance,
    'offset_xy': _calculate_offset_xy, 
    'offset_xyz': _calculate_offset_xyz,
    'angle_from_axis': _calculate_angle_from_axis,
    'intersect_two_planes': _calculate_intersect_two_planes,
    'tangent_to_cylinder': _calculate_tangent_to_cylinder
}

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
    
    def __init__(self, toltype:ToleranceType, value:float, negvalue:float=None):
        
        self.toltype=toltype
        self.pos=value
        self.neg=negvalue
    
    def __repr__(self):
        if self.neg:
            return f"Tolerance({self.toltype.name.capitalize()},+{self.pos} -{self.neg})"
        else:
            return f"Tolerance({self.toltype.name.capitalize()},±{self.pos})"
    
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
        