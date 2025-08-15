from typing import Dict, List, TYPE_CHECKING, Optional, Union, Tuple
import string
import numpy as np
from collections import defaultdict, OrderedDict

from .datum import Datum, Dimension, ToleranceSet, Tolerance, ToleranceType, DOF
from .geometry import Point, Plane, Axis, Cylinder, GeometryBase, GeometryType

if TYPE_CHECKING:
    from .assembly import Assembly

class Component:
    """
    Represents a mechanical component with geometries and datums.
    
    The component manages:
    - Geometries: The pure geometric entities (planes, axes, cylinders, etc.)
    - Datums: References to geometries used in GD&T
    - Dimensions: Relationships between geometries (parent-child)
    - Tolerances: Form, orientation, and position tolerances
    - Form Errors: Basis functions for modeling geometric deviations
    """
    
    def __init__(self, name: str, initial_geometry: GeometryBase = None):
        self.name = name
        
        # Default to top plane if no initial geometry provided
        if not initial_geometry:
            initial_geometry = Plane(origin=[0,0,0], u=[0,0,1])
        
        # Core data structures
        self.geometries: OrderedDict[str, GeometryBase] = OrderedDict()
        self.datums: OrderedDict[str, str] = OrderedDict()  # Maps datum label (A, B, C) to geometry ID
        self.dimensions: Dict[str, Dimension] = {}  # Maps geometry ID to its dimension from reference
        self.tolerances: Dict[str, 'ToleranceSet'] = {}  # Maps geometry ID to its tolerances
        
        #these are kept in the geometry!
        # self.form_errors: Dict[str, 'FormError'] = {}  # Maps geometry ID to form error model
        
        # Relationship tracking
        self.parent_child: Dict[str, str] = {}  # Maps child geometry ID to parent geometry ID
        self.children: Dict[str, List[str]] = defaultdict(list)  # Maps parent ID to list of children
        self.datum_references: Dict[str, List[str]] = {}  # Maps geometry ID to datum reference frame
        
        # Primary datum and coordinate system
        self.primary_datum_id: Optional[str] = None
        self.is_positioned = False
        
        # Track geometry counts by type for ID generation
        self._geo_counters = defaultdict(int)
        self._datum_counter = 0
        
        # Add the initial geometry as primary datum
        init_id = self._generate_geometry_id(initial_geometry.geotype)
        self.add_geometry(init_id, initial_geometry, is_primary=True)
        
        # Assembly relationships (for later)
        self.parent: Optional['Assembly'] = None
    
    def _generate_geometry_id(self, geo_type: GeometryType) -> str:
        """Generate an ID for a geometry based on its type (e.g., PlaneA, AxisB)."""
        type_name = geo_type.name.capitalize()
        
        # Get the next letter for this geometry type
        counter = self._geo_counters[type_name]
        if counter >= 26:
            # Use double letters if we run out (AA, AB, etc.)
            letter = string.ascii_uppercase[counter // 26 - 1] + string.ascii_uppercase[counter % 26]
        else:
            letter = string.ascii_uppercase[counter]
        
        self._geo_counters[type_name] += 1
        return f"{type_name}{letter}"
    
    def _generate_datum_label(self) -> str:
        """Generate next available datum label (A, B, C, ...)."""
        if self._datum_counter >= 26:
            # Use double letters if needed
            label = string.ascii_uppercase[self._datum_counter // 26 - 1] + string.ascii_uppercase[self._datum_counter % 26]
        else:
            label = string.ascii_uppercase[self._datum_counter]
        self._datum_counter += 1
        return label
    
    def add_geometry(self, geo_id: str, geometry: GeometryBase, 
                    reference_id: Optional[str] = None,
                    datum_label: Optional[str] = None,
                    is_primary: bool = False,
                    **dim_params) -> str:
        """
        Add a geometry to the component with proper relationship tracking.
        
        Args:
            geo_id: Unique identifier for the geometry
            geometry: The geometry object
            reference_id: ID of reference geometry (parent)
            datum_label: Optional datum label for GD&T (A, B, C, etc.)
            is_primary: Whether this is the primary datum
            **dim_params: Dimension parameters (dx, dy, dz, rx, ry, rz)
            
        Returns:
            The geometry ID
        """
        # Store the geometry (no references between geometries)
        self.geometries[geo_id] = geometry
        
        # Handle parent-child relationships
        if reference_id:
            if reference_id not in self.geometries:
                raise ValueError(f"Reference geometry '{reference_id}' not found")
            
            self.parent_child[geo_id] = reference_id
            self.children[reference_id].append(geo_id)
            
            # Create dimension from reference
            self.dimensions[geo_id] = Dimension(reference_geometry=self.geometries[reference_id])
            
            # Set dimension values if provided
            if any(k in dim_params for k in ['dx', 'dy', 'dz']):
                self.dimensions[geo_id].set_translation(
                    dx=dim_params.get('dx', 0),
                    dy=dim_params.get('dy', 0),
                    dz=dim_params.get('dz', 0)
                )
            if any(k in dim_params for k in ['rx', 'ry', 'rz']):
                self.dimensions[geo_id].set_rotation(
                    rx=dim_params.get('rx', 0),
                    ry=dim_params.get('ry', 0),
                    rz=dim_params.get('rz', 0)
                )
        else:
            # No parent - this is primary or floating
            self.dimensions[geo_id] = Dimension()
        
        # Initialize tolerance set for this geometry
        self.tolerances[geo_id] = ToleranceSet()
        
        # Initialize form error model (will be populated when tolerances are set)
        self.form_errors[geo_id] = None
        
        # Handle datum designation
        if datum_label or is_primary:
            if not datum_label:
                datum_label = self._generate_datum_label()
            self.datums[datum_label] = geo_id
            
            if is_primary:
                self.primary_datum_id = geo_id
        
        # Store datum reference frame if this references other datums
        if reference_id and reference_id in list(self.datums.values()):
            # Find all datum labels this references
            ref_datums = [label for label, gid in self.datums.items() 
                         if gid == reference_id or gid in self._get_ancestry(reference_id)]
            self.datum_references[geo_id] = ref_datums
        
        return geo_id
    
    def _get_ancestry(self, geo_id: str) -> List[str]:
        """Get all ancestor geometry IDs."""
        ancestors = []
        current = geo_id
        while current in self.parent_child:
            parent = self.parent_child[current]
            ancestors.append(parent)
            current = parent
        return ancestors
    
    def derive(self, reference_id: str, new_geo_type: Union[int, GeometryType], 
              geo_id: Optional[str] = None, **params) -> str:
        """
        Derive a new geometry from an existing one.
        
        Args:
            reference_id: ID of the reference geometry
            new_geo_type: Type of geometry to create
            geo_id: Optional ID for new geometry (auto-generated if not provided)
            **params: Derivation parameters (du, dv, dw, rx, ry, rz, datum_label, etc.)
            
        Returns:
            ID of the newly created geometry
        """
        if reference_id not in self.geometries:
            raise ValueError(f"Reference geometry '{reference_id}' not found")
        
        ref_geo = self.geometries[reference_id]
        
        # Extract datum_label if provided
        datum_label = params.pop('datum_label', None)
        
        # Use the geometry's derive method
        new_geo = ref_geo.derive(new_geo=new_geo_type, **params)
        
        # Generate ID if not provided
        if not geo_id:
            geo_id = self._generate_geometry_id(new_geo.geotype)
        
        # Extract dimension params
        dim_params = {}
        for key in ['du', 'dv', 'dw', 'dx', 'dy', 'dz', 'rx', 'ry', 'rz']:
            if key in params:
                # Map du->dx, dv->dy, dw->dz for consistency
                if key.startswith('d'):
                    dim_key = 'dx' if key == 'du' else ('dy' if key == 'dv' else ('dz' if key == 'dw' else key))
                else:
                    dim_key = key
                dim_params[dim_key] = params[key]
        
        # Add the new geometry with proper relationships
        self.add_geometry(geo_id, new_geo, reference_id=reference_id, 
                         datum_label=datum_label, **dim_params)
        
        return geo_id
    
    # Convenience methods for building components
    def add_plane(self, reference_id: Optional[str] = None, 
                 offset: float = 0, direction: Optional[np.ndarray] = None,
                 geo_id: Optional[str] = None, **kwargs) -> str:
        """Add a plane to the component."""
        if reference_id:
            if direction is None:
                # Default to normal direction of reference if it's a plane
                ref_geo = self.geometries[reference_id]
                if ref_geo.geotype == GeometryType.PLANE:
                    direction = ref_geo.u
                else:
                    direction = np.array([0, 0, 1])
            
            direction = direction / np.linalg.norm(direction)
            return self.derive(reference_id, GeometryType.PLANE, 
                             geo_id=geo_id, du=offset, **kwargs)
        else:
            # Create independent plane
            origin = kwargs.pop('origin', [0, 0, 0])
            normal = kwargs.pop('normal', [0, 0, 1])
            plane = Plane(origin=origin, u=normal)
            
            if not geo_id:
                geo_id = self._generate_geometry_id(plane.geotype)
            
            return self.add_geometry(geo_id, plane, **kwargs)
    
    def add_axis(self, reference_id: Optional[str] = None,
                origin: Optional[np.ndarray] = None,
                direction: Optional[np.ndarray] = None,
                geo_id: Optional[str] = None, **kwargs) -> str:
        """Add an axis to the component."""
        if reference_id:
            params = {}
            if origin is not None:
                params.update({'du': origin[0], 'dv': origin[1], 'dw': origin[2]})
            params.update(kwargs)
            return self.derive(reference_id, GeometryType.AXIS, geo_id=geo_id, **params)
        else:
            # Create independent axis
            origin = origin if origin is not None else [0, 0, 0]
            direction = direction if direction is not None else [0, 0, 1]
            axis = Axis(origin=origin, u=direction)
            
            if not geo_id:
                geo_id = self._generate_geometry_id(axis.geotype)
            
            return self.add_geometry(geo_id, axis, **kwargs)
    
    def add_cylinder(self, reference_id: Optional[str] = None,
                    origin: Optional[np.ndarray] = None,
                    direction: Optional[np.ndarray] = None,
                    radius: float = 1.0,
                    geo_id: Optional[str] = None, **kwargs) -> str:
        """Add a cylinder to the component."""
        if reference_id:
            params = {'r': radius}
            if origin is not None:
                params.update({'du': origin[0], 'dv': origin[1], 'dw': origin[2]})
            params.update(kwargs)
            return self.derive(reference_id, GeometryType.CYLINDER, geo_id=geo_id, **params)
        else:
            # Create independent cylinder
            origin = origin if origin is not None else [0, 0, 0]
            direction = direction if direction is not None else [0, 0, 1]
            cylinder = Cylinder(origin=origin, u=direction, r=radius)
            
            if not geo_id:
                geo_id = self._generate_geometry_id(cylinder.geotype)
            
            return self.add_geometry(geo_id, cylinder, **kwargs)
    
    def add_point(self, reference_id: Optional[str] = None,
                 position: Optional[np.ndarray] = None,
                 geo_id: Optional[str] = None, **kwargs) -> str:
        """Add a point to the component."""
        if reference_id:
            if position is not None:
                return self.derive(reference_id, GeometryType.POINT, geo_id=geo_id,
                                 du=position[0], dv=position[1], dw=position[2], **kwargs)
            else:
                return self.derive(reference_id, GeometryType.POINT, geo_id=geo_id, **kwargs)
        else:
            # Create independent point
            position = position if position is not None else [0, 0, 0]
            point = Point(origin=position)
            
            if not geo_id:
                geo_id = self._generate_geometry_id(point.geotype)
            
            return self.add_geometry(geo_id, point, **kwargs)
    
    def intersect_geometries(self, geo_id1: str, geo_id2: str, 
                           geo_id: Optional[str] = None) -> str:
        """
        Create new geometry from intersection of two existing geometries.
        
        Args:
            geo_id1: ID of first geometry
            geo_id2: ID of second geometry
            geo_id: Optional ID for intersection geometry
            
        Returns:
            ID of the intersection geometry
        """
        if geo_id1 not in self.geometries or geo_id2 not in self.geometries:
            raise ValueError(f"One or both geometries not found")
        
        geo1 = self.geometries[geo_id1]
        geo2 = self.geometries[geo_id2]
        
        # Calculate intersection
        intersection = geo1.intersection(geo2)
        
        if intersection.geotype == GeometryType.NULL:
            raise ValueError(f"No intersection between {geo_id1} and {geo_id2}")
        
        if not geo_id:
            geo_id = self._generate_geometry_id(intersection.geotype)
        
        # Add the intersection - it references both parent geometries conceptually
        # but we'll use the first as the primary reference
        return self.add_geometry(geo_id, intersection, reference_id=geo_id1)
    
    # Tolerance management
    def set_tolerance(self, geo_id: str, tol_type: ToleranceType, 
                     value: float, dof: Optional[DOF] = None):
        """
        Set a tolerance for a geometry.
        
        Args:
            geo_id: Geometry ID
            tol_type: Type of tolerance (FLATNESS, POSITION, etc.)
            value: Tolerance value
            dof: Degree of freedom (for position/orientation tolerances)
        """
        if geo_id not in self.tolerances:
            raise ValueError(f"Geometry '{geo_id}' not found")
        
        tol = Tolerance(tol_type, value)
        
        if dof is not None:
            # Position or orientation tolerance
            self.dimensions[geo_id].set_tolerance(dof, tol)
        else:
            # Form tolerance - store in tolerance set
            self.tolerances[geo_id].add_tolerance(tol_type, tol)
            
            # Initialize form error model if needed
            if tol_type in [ToleranceType.FLATNESS, ToleranceType.STRAIGHTNESS, 
                           ToleranceType.CYLINDRICITY, ToleranceType.CIRCULARITY]:
                self._initialize_form_error(geo_id, tol_type, value)
    
    def _initialize_form_error(self, geo_id: str, tol_type: ToleranceType, tol_value: float):
        """Initialize form error model for a geometry."""
        # This will be expanded to use proper basis functions
        # For now, just store the tolerance value as a placeholder
        self.form_errors[geo_id] = {
            'type': tol_type,
            'tolerance': tol_value,
            'std_dev': tol_value / 3,  # Assuming 3-sigma for tolerance zone
            'basis_functions': None  # To be implemented
        }
    
    # Query methods
    def get_geometry(self, geo_id: str) -> GeometryBase:
        """Get a geometry by ID."""
        if geo_id not in self.geometries:
            raise ValueError(f"Geometry '{geo_id}' not found")
        return self.geometries[geo_id]
    
    def get_datum_geometry(self, datum_label: str) -> GeometryBase:
        """Get geometry associated with a datum label."""
        if datum_label not in self.datums:
            raise ValueError(f"Datum '{datum_label}' not found")
        return self.geometries[self.datums[datum_label]]
    
    def get_children(self, geo_id: str) -> List[str]:
        """Get all child geometries of a given geometry."""
        return self.children.get(geo_id, [])
    
    def get_parent(self, geo_id: str) -> Optional[str]:
        """Get parent geometry of a given geometry."""
        return self.parent_child.get(geo_id)
    
    def get_datum_label(self, geo_id: str) -> Optional[str]:
        """Get datum label for a geometry if it has one."""
        for label, gid in self.datums.items():
            if gid == geo_id:
                return label
        return None
    
    def list_geometries(self) -> Dict[str, str]:
        """List all geometries with their types."""
        return {
            geo_id: geo.geotype.name
            for geo_id, geo in self.geometries.items()
        }
    
    def list_datums(self) -> Dict[str, str]:
        """List all datums with their geometry types."""
        return {
            label: f"{self.geometries[geo_id].geotype.name} ({geo_id})"
            for label, geo_id in self.datums.items()
        }
    
    def get_tolerance_summary(self, geo_id: str) -> Dict:
        """Get summary of all tolerances for a geometry."""
        if geo_id not in self.geometries:
            raise ValueError(f"Geometry '{geo_id}' not found")
        
        summary = {
            'geometry_type': self.geometries[geo_id].geotype.name,
            'datum_label': self.get_datum_label(geo_id),
            'parent': self.get_parent(geo_id),
            'children': self.get_children(geo_id),
            'tolerances': {}
        }
        
        # Add position/orientation tolerances from dimension
        if geo_id in self.dimensions and self.dimensions[geo_id]:
            dim = self.dimensions[geo_id]
            for dof, tol in dim.tolerances.items():
                if tol.pos > 0 or (tol.neg and tol.neg > 0):
                    summary['tolerances'][dof.name] = str(tol)
        
        # Add form tolerances
        if geo_id in self.tolerances:
            tol_set = self.tolerances[geo_id]
            # This assumes ToleranceSet has a method to list tolerances
            # We'll need to implement this
        
        return summary
    
    def __getattr__(self, attr: str):
        """Allow accessing geometries as attributes (e.g., component.PlaneA)."""
        if attr in self.geometries:
            return self.geometries[attr]
        # Also allow datum access (e.g., component.A for datum A)
        if attr in self.datums:
            return self.geometries[self.datums[attr]]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")
    
    def print_tree(self, show_tolerances: bool = False, show_dimensions: bool = False):
        """Print a nicely formatted tree of the component structure."""
        print(f"\n{'='*60}")
        print(f"Component: {self.name}")
        print(f"{'='*60}")
        
        # Print primary datum
        if self.primary_datum_id:
            print(f"\n📍 Primary Datum:")
            self._print_geometry_node(self.primary_datum_id, prefix="  ", 
                                     show_tolerances=show_tolerances,
                                     show_dimensions=show_dimensions)
        
        # Print geometry tree
        print(f"\n🌲 Geometry Tree:")
        # Find root geometries (those without parents)
        roots = [geo_id for geo_id in self.geometries 
                if geo_id not in self.parent_child]
        
        for root_id in roots:
            self._print_geometry_tree(root_id, indent="  ", 
                                     show_tolerances=show_tolerances,
                                     show_dimensions=show_dimensions)
        
        # Print datum assignments
        if self.datums:
            print(f"\n📐 Datum Assignments:")
            for label, geo_id in self.datums.items():
                geo = self.geometries[geo_id]
                print(f"  Datum {label} → {geo_id} ({geo.geotype.name})")
        
        print(f"\n{'='*60}\n")
    
    def _print_geometry_tree(self, geo_id: str, indent: str = "", 
                            show_tolerances: bool = False,
                            show_dimensions: bool = False,
                            is_last: bool = True):
        """Recursively print geometry tree."""
        # Choose the connector symbol
        connector = "└── " if is_last else "├── "
        
        # Print this node
        self._print_geometry_node(geo_id, indent + connector,
                                 show_tolerances=show_tolerances,
                                 show_dimensions=show_dimensions)
        
        # Get children
        children = self.get_children(geo_id)
        
        # Prepare indent for children
        extension = "    " if is_last else "│   "
        child_indent = indent + extension
        
        # Print children
        for i, child_id in enumerate(children):
            is_last_child = (i == len(children) - 1)
            self._print_geometry_tree(child_id, child_indent,
                                     show_tolerances=show_tolerances,
                                     show_dimensions=show_dimensions,
                                     is_last=is_last_child)
    
    def _print_geometry_node(self, geo_id: str, prefix: str = "",
                            show_tolerances: bool = False,
                            show_dimensions: bool = False):
        """Print a single geometry node with its information."""
        geo = self.geometries[geo_id]
        datum_label = self.get_datum_label(geo_id)
        
        # Build the node string
        node_str = f"{geo_id} ({geo.geotype.name})"
        
        # Add datum label if exists
        if datum_label:
            node_str += f" [Datum {datum_label}]"
        
        # Add geometry details
        if hasattr(geo, 'origin'):
            origin_str = f"[{geo.origin[0]:.1f}, {geo.origin[1]:.1f}, {geo.origin[2]:.1f}]"
            node_str += f" @ {origin_str}"
        
        if hasattr(geo, 'r') and geo.r > 0:
            node_str += f" r={geo.r:.2f}"
        
        print(prefix + node_str)
        
        # Show dimensions if requested
        if show_dimensions and geo_id in self.dimensions and self.dimensions[geo_id]:
            dim = self.dimensions[geo_id]
            if dim.reference_geometry:
                trans = dim.get_translation_vector()
                rot = dim.get_rotation_vector()
                if np.any(trans != 0) or np.any(rot != 0):
                    dim_prefix = prefix.replace("├──", "│  ").replace("└──", "   ")
                    print(f"{dim_prefix}  ↳ Δ: [{trans[0]:.1f}, {trans[1]:.1f}, {trans[2]:.1f}]",
                          end="")
                    if np.any(rot != 0):
                        print(f" θ: [{np.degrees(rot[0]):.1f}°, {np.degrees(rot[1]):.1f}°, {np.degrees(rot[2]):.1f}°]")
                    else:
                        print()
        
        # Show tolerances if requested
        if show_tolerances and geo_id in self.tolerances:
            tol_set = self.tolerances[geo_id]
            tol_strs = []
            
            # Form tolerances
            for tol_type, tol in tol_set.form_tolerances.items():
                tol_strs.append(f"{tol_type.name}: ±{tol.pos}")
            
            # Position tolerances
            for dof, tol in tol_set.position_tolerances.items():
                tol_strs.append(f"POS_{dof.name}: ±{tol.pos}")
            
            if tol_strs:
                tol_prefix = prefix.replace("├──", "│  ").replace("└──", "   ")
                print(f"{tol_prefix}  ⚙ Tol: {', '.join(tol_strs)}")
    
    def display(self):
        """Display comprehensive component information."""
        self.print_tree(show_tolerances=True, show_dimensions=True)
    
    def __repr__(self):
        geo_count = len(self.geometries)
        datum_count = len(self.datums)
        positioned = "positioned" if self.is_positioned else "floating"
        
        return f"Component('{self.name}', {geo_count} geometries, {datum_count} datums, {positioned})"

