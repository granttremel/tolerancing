
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

from .geometry.geometry import GeometryType, GeometryBase

if TYPE_CHECKING:
    from .component import Component
    from .geometry import GeometryBase


class DOF(Enum):
    """Degrees of freedom for dimensions."""
    DX = 0  # Translation in X (u direction)
    DY = 1  # Translation in Y (v direction)
    DZ = 2  # Translation in Z (w direction)
    RX = 3  # Rotation about X (u axis)
    RY = 4  # Rotation about Y (v axis)
    RZ = 5  # Rotation about Z (w axis)


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

class ToleranceSet:
    """Container for managing multiple tolerances on a geometry."""
    
    def __init__(self):        
        self.position_tolerances: Dict[DOF, Tolerance] = {}
        self.orientation_tolerances: Dict[Tuple[ToleranceType, DOF], Tolerance] = {}
        self.form_tolerances: Dict[ToleranceType, Tolerance] = {}

    def add_tolerance(self, tol_type: ToleranceType, tolerance: Tolerance, dof: Optional[DOF] = None):
        """Add a tolerance to the set."""
        if tol_type in [ToleranceType.FLATNESS, ToleranceType.STRAIGHTNESS,
                        ToleranceType.CYLINDRICITY, ToleranceType.CIRCULARITY]:
            self.form_tolerances[tol_type] = tolerance
        elif tol_type in [ToleranceType.PERPENDICULARITY, ToleranceType.PARALLELISM,
                         ToleranceType.ANGULARITY]:
            if dof is None:
                raise ValueError(f"DOF required for {tol_type.name} tolerance")
            self.orientation_tolerances[(tol_type, dof)] = tolerance
        elif tol_type == ToleranceType.POSITION:
            if dof is None:
                raise ValueError("DOF required for position tolerance")
            self.position_tolerances[dof] = tolerance
        else:
            # General tolerance
            self.form_tolerances[tol_type] = tolerance
    
    def get_form_tolerances(self) -> Dict[ToleranceType, Tolerance]:
        """Get all form tolerances."""
        return self.form_tolerances.copy()
    
    def get_orientation_tolerances(self) -> Dict[Tuple[ToleranceType, DOF], Tolerance]:
        """Get all orientation tolerances."""
        return self.orientation_tolerances.copy()
    
    def get_position_tolerances(self) -> Dict[DOF, Tolerance]:
        """Get all position tolerances."""
        return self.position_tolerances.copy()
    
class ToleranceSet:
    """Manages all tolerances for a single geometry"""
    
    def __init__(self):
        self.position = {}     # X, Y, Z position tolerances
        self.orientation = {}  # Angular tolerances
        self.form = {}        # Flatness, straightness, etc.
        self.runout = {}      # Circular/total runout
        
    def add_position_tolerance(self, dof: DOF, tol: Tolerance):
        self.position[dof] = tol
        
    def add_form_tolerance(self, tol_type: ToleranceType, 
                          value: float, modal_dist='default'):
        self.form[tol_type] = {
            'value': value,
            'modal_distribution': modal_dist
        }
    
    def get_modal_weights(self) -> dict:
        """Convert form tolerances to modal weights"""
        if not self.form:
            return {}
            
        # Combine multiple form tolerances
        weights = {}
        for tol_type, spec in self.form.items():
            dist = self._get_distribution(spec['modal_distribution'])
            
            # Scale by tolerance value
            scale = spec['value'] / 3  # 3-sigma
            for mode, weight in dist.items():
                if mode in weights:
                    # RSS combination if multiple tolerances affect same mode
                    weights[mode] = np.sqrt(weights[mode]**2 + (weight*scale)**2)
                else:
                    weights[mode] = weight * scale
        
        return weights

class Datum:
    
    def __init__(self, geo: GeometryBase, dimension: 'Dimension' = None, name: str = None):
        """
        Create a datum from geometry with optional dimension relationship.
        
        Args:
            geo: The geometry for this datum
            dimension: Optional dimension relationship from reference geometry
            name: Optional name for the datum
        """
        self.name = name
        self.geo = geo
        self.dimension = dimension if dimension else Dimension(reference_geometry=geo)
        self.fixed = False
        self.reference = None
        self.parent = None
    
    @classmethod
    def from_geometry(cls, geometry: GeometryBase, name: str = None) -> 'Datum':
        """Create a primary datum from geometry"""
        return cls(geo=geometry, name=name)
    
    def set_reference(self, reference: 'Datum'):
        self.reference = reference
        # Only set reference if geo has the method (for backwards compatibility)
        if hasattr(self.geo, 'set_reference'):
            self.geo.set_reference(reference.geo)
    
    def set_parent(self, component: 'Component'):
        self.parent = component
        
    def set_tolerance(self, dof: DOF, tolerance: 'Tolerance'):
        """Set tolerance for a specific degree of freedom"""
        self.dimension.set_tolerance(dof, tolerance)
        
    def is_defined(self) -> bool:
        """Check if datum is fully geometrically defined"""
        return self.geo is not None
    
    def get_construction_chain(self) -> List['Datum']:
        """Get the chain of datums this datum depends on"""
        if not self.dimension or not self.dimension.reference_geometry:
            return [self]  # Primary datum
        
        # For now, just return self since we don't track datum chains the same way
        return [self]
    
    
    def __repr__(self):
        geo_str = f"{self.geo.geotype.name}" if self.geo else "undefined"
        name_str = f"'{self.name}'" if self.name else "unnamed"
        return f"Datum({name_str}, {geo_str})"


class Dimension:
    """
    Represents a 6-DOF dimensional relationship from a reference geometry.
    Each degree of freedom has an associated tolerance.
    """
    
    _dimension_counter = 0  # Class variable to track dimension count
    
    def __init__(self, reference_geometry: 'GeometryBase' = None, name: str = None):
        """
        reference_geometry: The geometry this dimension references
        name: Optional name for the dimension
        """
        # Increment counter and assign ID
        Dimension._dimension_counter += 1
        self.id = f"DIM_{Dimension._dimension_counter:04d}"
        
        # Generate name if not provided
        if name:
            self.name = name
        else:
            self.name = f"dimension_{self.id}"
        
        self.reference_geometry = reference_geometry
        
        # Initialize 6 DOF values (in reference geometry's uvw coordinate system)
        # These represent the transformation from reference to this geometry
        self.values = {
            DOF.DX: 0.0,  # u translation
            DOF.DY: 0.0,  # v translation
            DOF.DZ: 0.0,  # w translation
            DOF.RX: 0.0,  # u rotation (radians)
            DOF.RY: 0.0,  # v rotation (radians)
            DOF.RZ: 0.0,  # w rotation (radians)
        }
        
        # Initialize tolerances for each DOF (default to ±0)
        self.tolerances = {
            DOF.DX: Tolerance(ToleranceType.DIMENSION, 0.0),
            DOF.DY: Tolerance(ToleranceType.DIMENSION, 0.0),
            DOF.DZ: Tolerance(ToleranceType.DIMENSION, 0.0),
            DOF.RX: Tolerance(ToleranceType.ANGULARITY, 0.0),
            DOF.RY: Tolerance(ToleranceType.ANGULARITY, 0.0),
            DOF.RZ: Tolerance(ToleranceType.ANGULARITY, 0.0),
        }
    
    def set_dof(self, dof: DOF, value: float, tolerance: Tolerance = None):
        """Set a specific degree of freedom value and optionally its tolerance."""
        self.values[dof] = value
        if tolerance:
            self.tolerances[dof] = tolerance
    
    def set_translation(self, dx: float = None, dy: float = None, dz: float = None):
        """Set translation values."""
        if dx is not None:
            self.values[DOF.DX] = dx
        if dy is not None:
            self.values[DOF.DY] = dy
        if dz is not None:
            self.values[DOF.DZ] = dz
    
    def set_rotation(self, rx: float = None, ry: float = None, rz: float = None):
        """Set rotation values (in radians)."""
        if rx is not None:
            self.values[DOF.RX] = rx
        if ry is not None:
            self.values[DOF.RY] = ry
        if rz is not None:
            self.values[DOF.RZ] = rz
    
    def set_tolerance(self, dof: DOF, tolerance: Tolerance):
        """Set tolerance for a specific degree of freedom."""
        self.tolerances[dof] = tolerance
    
    def get_translation_vector(self) -> np.ndarray:
        """Get the translation vector [dx, dy, dz]."""
        return np.array([self.values[DOF.DX], self.values[DOF.DY], self.values[DOF.DZ]])
    
    def get_rotation_vector(self) -> np.ndarray:
        """Get the rotation vector [rx, ry, rz]."""
        return np.array([self.values[DOF.RX], self.values[DOF.RY], self.values[DOF.RZ]])
    
    def __repr__(self):
        trans = self.get_translation_vector()
        rot = self.get_rotation_vector()
        return f"Dimension({self.name}, trans={trans}, rot={rot})"

