
from typing import Dict, List, TYPE_CHECKING, Optional, Union
import string

from .datum import Datum, Relation
from .geometry import GeometryType

if TYPE_CHECKING:
    from .assembly import Assembly


class Component:
    """
    Represents a mechanical component with datums that define its geometry.
    Components can be mated together in assemblies through their datums.
    """
    
    def __init__(self, name: str):
        self.name = name
        
        # Datum management
        self.primary_datum: Optional[Datum] = None
        self.datums: Dict[str, Datum] = {}
        self.relations: List[Relation] = []  # Fixed initialization
        
        # Assembly relationships
        self.mating: List = []
        self.parent: Optional['Assembly'] = None
        
        # Component coordinate system (updated when mated)
        self.position = [0, 0, 0]
        self.orientation = [0, 0, 0]
        self.is_positioned = False
    
    def set_parent(self, assy: 'Assembly') -> None:
        """Set the parent assembly for this component"""
        self.parent = assy
    
    def add_datum(self, datum: Datum, id: str = "") -> str:
        """
        Add a datum to this component.
        
        Args:
            datum: The datum to add
            id: Optional ID for the datum (auto-generated if empty)
            
        Returns:
            The ID assigned to the datum
        """
        if not id:
            # Auto-generate alphabetic ID
            id = string.ascii_uppercase[len(self.datums)]
            
        # Set up datum relationships
        if not self.primary_datum:
            self.primary_datum = datum
            datum.fixed = True  # Primary datum is fixed
        else:
            # Secondary datums reference the primary
            datum.set_reference(self.primary_datum)
            
        self.datums[id] = datum
        datum.set_parent(self)
        
        return id
    
    def add_primary_datums(self, plane_origin=[0,0,0], plane_normal=[0,0,1], 
                          axis_origin=[0,0,0], axis_direction=[0,0,1]) -> tuple:
        """
        Convenience method to add standard primary datums (plane A, axis B).
        
        Returns:
            Tuple of (plane_datum_id, axis_datum_id)
        """
        # Add primary plane datum
        plane_datum = Datum.from_geometry(
            GeometryType.PLANE, plane_origin, plane_normal, name=f"{self.name}_A"
        )
        plane_id = self.add_datum(plane_datum)
        
        # Add primary axis datum
        axis_datum = Datum.from_geometry(
            GeometryType.AXIS, axis_origin, axis_direction, name=f"{self.name}_B" 
        )
        axis_id = self.add_datum(axis_datum)
        
        return plane_id, axis_id
    
    def create_derived_datum(self, reference_id: str, dimension_type: str,
                           values: List[float], target_geometry: GeometryType,
                           constraints: Dict = None, datum_id: str = "",
                           name: str = None) -> str:
        """
        Create a derived datum from an existing datum in this component.
        
        Args:
            reference_id: ID of the reference datum in this component
            dimension_type: Type of dimensional relationship
            values: Dimensional values
            target_geometry: Target geometry type
            constraints: Additional constraints
            datum_id: Optional ID for new datum
            name: Optional name for new datum
            
        Returns:
            ID of the created datum
        """
        if reference_id not in self.datums:
            raise ValueError(f"Reference datum '{reference_id}' not found in component '{self.name}'")
        
        ref_datum = self.datums[reference_id]
        
        new_datum = Datum.from_dimension(
            reference_datums=[ref_datum],
            dimension_type=dimension_type,
            values=values,
            target_geometry=target_geometry,
            constraints=constraints,
            name=name or f"{self.name}_{datum_id or len(self.datums)}"
        )
        
        return self.add_datum(new_datum, datum_id)
    
    def get_datum(self, id: str) -> Datum:
        """Get a datum by ID"""
        if id not in self.datums:
            raise ValueError(f"Datum '{id}' not found in component '{self.name}'")
        return self.datums[id]
    
    def list_datums(self) -> Dict[str, str]:
        """Return a dictionary of datum IDs and their descriptions"""
        return {
            id: f"{datum.name or 'unnamed'} ({datum.geo.geotype.name})"
            for id, datum in self.datums.items()
        }
    
    def is_fully_defined(self) -> bool:
        """Check if all datums in this component are fully defined"""
        return all(datum.is_defined() for datum in self.datums.values())
    
    def get_mating_datums(self) -> List[str]:
        """Get list of datum IDs suitable for mating (typically geometric features)"""
        mating_datums = []
        for id, datum in self.datums.items():
            geo_type = datum.geo.geotype
            if geo_type in [GeometryType.PLANE, GeometryType.AXIS, GeometryType.CYLINDER, 
                           GeometryType.SPHERE, GeometryType.POINT]:
                mating_datums.append(id)
        return mating_datums
    
    def __getattr__(self, attr: str):
        """Allow accessing datums as attributes (e.g., component.A)"""
        if attr in self.datums:
            return self.datums[attr]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")
    
    def __repr__(self):
        datum_count = len(self.datums)
        primary = f"primary={self.primary_datum.name}" if self.primary_datum else "no primary"
        positioned = "positioned" if self.is_positioned else "floating"
        
        return f"Component('{self.name}', {datum_count} datums, {primary}, {positioned})"
        
