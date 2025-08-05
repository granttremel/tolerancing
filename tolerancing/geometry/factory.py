# Geometry factory to avoid circular imports
from typing import Dict, Type, TYPE_CHECKING
from .geometry import GeometryBase, GeometryType

if TYPE_CHECKING:
    from .point import Point
    from .axis import Axis
    from .plane import Plane

# Registry will be populated by each geometry class
_geometry_registry: Dict[GeometryType, Type[GeometryBase]] = {}

def register_geometry(geo_type: GeometryType, geo_class: Type[GeometryBase]):
    """Register a geometry class with its type"""
    _geometry_registry[geo_type] = geo_class

def create_geometry(geo_type: GeometryType, **params) -> GeometryBase:
    """Create a geometry instance by type"""
    if geo_type not in _geometry_registry:
        raise ValueError(f"Geometry type {geo_type} not registered")
    
    return _geometry_registry[geo_type](**params)

def get_geometry_class(geo_type: GeometryType) -> Type[GeometryBase]:
    """Get geometry class by type"""
    if geo_type not in _geometry_registry:
        # Lazy import if not yet registered
        if geo_type == GeometryType.POINT:
            from .point import Point
            return Point
        elif geo_type == GeometryType.AXIS:
            from .axis import Axis
            return Axis
        elif geo_type == GeometryType.PLANE:
            from .plane import Plane
            return Plane
        # Add more as needed
        else:
            raise ValueError(f"Unknown geometry type {geo_type}")
    
    return _geometry_registry[geo_type]