

from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import numpy as np
from dataclasses import dataclass, field

from .component import Component
from .datum import Datum
from .geometry.geometry import GeometryType
from .transformations import CoordinateTransformer, TransformationStep

class MateType(Enum):
    """Types of mating relationships between components"""
    COINCIDENT = "coincident"      # Two planes/points are coincident
    CONCENTRIC = "concentric"      # Two axes/cylinders are concentric
    PARALLEL = "parallel"          # Two planes/axes are parallel
    PERPENDICULAR = "perpendicular" # Two planes/axes are perpendicular
    TANGENT = "tangent"           # Surfaces are tangent

class Mate:
    """
    Represents a mating relationship between datums on two components.
    """
    
    _mate_counter=0
    
    def __init__(self, component1: Component, datum1_id: str, 
                 component2: Component, datum2_id: str, 
                 mate_type: MateType, offset: float = 0.0):
        """
        Create a mate between two component datums.
        
        Args:
            component1: First component
            datum1_id: ID of datum on first component
            component2: Second component  
            datum2_id: ID of datum on second component
            mate_type: Type of mating relationship
            offset: Optional offset distance for the mate
        """
        self.id = self._mate_counter
        self.component1 = component1
        self.component2 = component2
        self.datum1_id = datum1_id
        self.datum2_id = datum2_id
        self.mate_type = mate_type
        self.offset = offset
        
        self._mate_counter+=1
        # Validate the mate is geometrically possible
        self._validate_mate()
        
    def _validate_mate(self):
        """Validate that this mate is geometrically valid"""
        datum1 = self.component1.get_datum(self.datum1_id)
        datum2 = self.component2.get_datum(self.datum2_id)
        
        if not self._is_mate_valid(datum1, datum2, self.mate_type):
            raise ValueError(
                f"Invalid {self.mate_type.value} mate between "
                f"{datum1.geo.geotype.name} and {datum2.geo.geotype.name}"
            )
    
    @staticmethod
    def _is_mate_valid(datum1: Datum, datum2: Datum, mate_type: MateType) -> bool:
        """Check if a mate type is valid between two datum geometries"""
        geo1, geo2 = datum1.geo.geotype, datum2.geo.geotype
        
        if mate_type == MateType.COINCIDENT:
            # Planes can be coincident, points can be coincident
            return (geo1 == geo2 == GeometryType.PLANE or 
                   geo1 == geo2 == GeometryType.POINT)
                   
        elif mate_type == MateType.CONCENTRIC:
            # Axes can be concentric, cylinders can be concentric
            valid_pairs = [
                (GeometryType.AXIS, GeometryType.AXIS),
                (GeometryType.CYLINDER, GeometryType.CYLINDER),
                (GeometryType.AXIS, GeometryType.CYLINDER),
                (GeometryType.CYLINDER, GeometryType.AXIS)
            ]
            return (geo1, geo2) in valid_pairs or (geo2, geo1) in valid_pairs
            
        elif mate_type == MateType.PARALLEL:
            # Planes can be parallel, axes can be parallel
            return ((geo1 in [GeometryType.PLANE, GeometryType.AXIS]) and
                   (geo2 in [GeometryType.PLANE, GeometryType.AXIS]))
                   
        elif mate_type == MateType.PERPENDICULAR:
            # Planes can be perpendicular, axes can be perpendicular to planes
            return ((geo1 in [GeometryType.PLANE, GeometryType.AXIS]) and
                   (geo2 in [GeometryType.PLANE, GeometryType.AXIS]))
                   
        elif mate_type == MateType.TANGENT:
            # Various tangent combinations possible
            tangent_pairs = [
                (GeometryType.PLANE, GeometryType.CYLINDER),
                (GeometryType.PLANE, GeometryType.SPHERE),
                (GeometryType.CYLINDER, GeometryType.SPHERE)
            ]
            return (geo1, geo2) in tangent_pairs or (geo2, geo1) in tangent_pairs
        
        return False
    
    def get_datums(self) -> Tuple[Datum, Datum]:
        """Get the two datums involved in this mate"""
        datum1 = self.component1.get_datum(self.datum1_id)
        datum2 = self.component2.get_datum(self.datum2_id)
        return datum1, datum2
    
    def __repr__(self):
        return (f"Mate({self.component1.name}.{self.datum1_id} {self.mate_type.value} "
               f"{self.component2.name}.{self.datum2_id})")


class Assembly:
    """
    Container class for components that manages mating relationships and positioning.
    """
    
    def __init__(self, name: str = "Assembly"):
        self.name = name
        self.components: Dict[str, Component] = {}
        self.mates: List[Mate] = []
        
        # First component added becomes the "ground" (fixed reference)
        self.ground_component: Optional[Component] = None
        
        # Coordinate transformation system
        self.transformer = CoordinateTransformer()
    
    def add_component(self, component: Component, ground: bool = False) -> None:
        """
        Add a component to the assembly.
        
        Args:
            component: Component to add
            ground: If True, this component becomes the fixed ground reference
        """
        if component.name in self.components:
            raise ValueError(f"Component '{component.name}' already exists in assembly")
        
        self.components[component.name] = component
        component.set_parent(self)
        
        # First component or explicitly grounded component becomes ground
        if self.ground_component is None or ground:
            self.ground_component = component
            component.is_positioned = True
            print(f"Component '{component.name}' set as ground reference")
    
    def mate_components(self, comp1_name: str, datum1_id: str,
                       comp2_name: str, datum2_id: str,
                       mate_type: MateType, offset: float = 0.0) -> Mate:
        """
        Create a mate between datums on two components.
        
        Args:
            comp1_name: Name of first component
            datum1_id: ID of datum on first component
            comp2_name: Name of second component
            datum2_id: ID of datum on second component
            mate_type: Type of mate
            offset: Optional offset distance
            
        Returns:
            The created mate
        """
        if comp1_name not in self.components:
            raise ValueError(f"Component '{comp1_name}' not found in assembly")
        if comp2_name not in self.components:
            raise ValueError(f"Component '{comp2_name}' not found in assembly")
        
        comp1 = self.components[comp1_name]
        comp2 = self.components[comp2_name]
        
        # Create and validate the mate
        mate = Mate(comp1, datum1_id, comp2, datum2_id, mate_type, offset)
        self.mates.append(mate)
        
        # Update positioning - if one component is positioned, position the other
        if comp1.is_positioned and not comp2.is_positioned:
            self._position_component_from_mate(comp2, comp1, mate)
        elif comp2.is_positioned and not comp1.is_positioned:
            self._position_component_from_mate(comp1, comp2, mate)
        
        print(f"Created mate: {mate}")
        return mate
    
    def _position_component_from_mate(self, moving_comp: Component, 
                                    fixed_comp: Component, mate: Mate):
        """Position a component based on a mate with a fixed component"""
        # This is a simplified positioning - in a full implementation,
        # you'd solve the constraint system to determine exact positioning
        
        moving_comp.is_positioned = True
        
        # Get the datums involved
        if mate.component1 == fixed_comp:
            fixed_datum = fixed_comp.get_datum(mate.datum1_id)
            moving_datum = moving_comp.get_datum(mate.datum2_id)
        else:
            fixed_datum = fixed_comp.get_datum(mate.datum2_id)
            moving_datum = moving_comp.get_datum(mate.datum1_id)
        
        # Simple positioning based on mate type
        if mate.mate_type == MateType.COINCIDENT:
            # Position moving component so datums are coincident
            offset_vector = fixed_datum.geo.aorigin - moving_datum.geo.aorigin
            moving_comp.position = offset_vector.tolist()
            
        elif mate.mate_type == MateType.CONCENTRIC:
            # Position moving component so axes are concentric
            offset_vector = fixed_datum.geo.aorigin - moving_datum.geo.aorigin
            moving_comp.position = offset_vector.tolist()
        
        print(f"Positioned component '{moving_comp.name}' based on mate")
    
    def get_component(self, name: str) -> Component:
        """Get a component by name"""
        if name not in self.components:
            raise ValueError(f"Component '{name}' not found in assembly")
        return self.components[name]
    
    def list_components(self) -> Dict[str, str]:
        """List all components with their status"""
        return {
            name: f"{comp} ({'ground' if comp == self.ground_component else 'floating'})"
            for name, comp in self.components.items()
        }
    
    def list_mates(self) -> List[str]:
        """List all mates in the assembly"""
        return [str(mate) for mate in self.mates]
    
    def is_fully_constrained(self) -> bool:
        """Check if all components are properly positioned"""
        return all(comp.is_positioned for comp in self.components.values())
    
    def component_is_defined(self, comp: Component) -> bool:
        """Check if a component is fully defined (all datums valid)"""
        return comp.is_fully_defined()
    
    def validate_assembly(self) -> Dict[str, List[str]]:
        """Validate the entire assembly and return any issues"""
        issues = {
            'undefined_components': [],
            'unpositioned_components': [],
            'invalid_mates': []
        }
        
        for name, comp in self.components.items():
            if not self.component_is_defined(comp):
                issues['undefined_components'].append(name)
            if not comp.is_positioned:
                issues['unpositioned_components'].append(name)
        
        # Check for mate issues (this could be expanded)
        for mate in self.mates:
            try:
                mate._validate_mate()
            except ValueError as e:
                issues['invalid_mates'].append(f"{mate}: {e}")
        
        return issues
    
    def get_datum_transfer_map(self) -> Dict[str, List[str]]:
        """
        Generate a map showing how datums transfer between components through mates.
        This is useful for tolerance stackup analysis.
        """
        transfer_map = {}
        
        for mate in self.mates:
            comp1_key = f"{mate.component1.name}.{mate.datum1_id}"
            comp2_key = f"{mate.component2.name}.{mate.datum2_id}"
            
            if comp1_key not in transfer_map:
                transfer_map[comp1_key] = []
            if comp2_key not in transfer_map:
                transfer_map[comp2_key] = []
            
            transfer_map[comp1_key].append(comp2_key)
            transfer_map[comp2_key].append(comp1_key)
        
        return transfer_map
    
    def __getattr__(self, attr: str):
        """Allow accessing components as attributes"""
        if attr in self.components:
            return self.components[attr]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")
    
    def __repr__(self):
        comp_count = len(self.components)
        mate_count = len(self.mates)
        constrained = "fully constrained" if self.is_fully_constrained() else "under-constrained"
        
        return f"Assembly('{self.name}', {comp_count} components, {mate_count} mates, {constrained})"
    
    # ===== COORDINATE TRANSFORMATION METHODS =====
    
    def get_datum_global_transform(self, component_name: str, datum_id: str) -> List[TransformationStep]:
        """
        Get the complete transformation chain from a datum to global coordinates.
        
        Args:
            component_name: Name of the component
            datum_id: ID of the datum within the component
            
        Returns:
            List of transformation steps from datum to global coordinates
        """
        if component_name not in self.components:
            raise ValueError(f"Component '{component_name}' not found in assembly")
        
        component = self.components[component_name]
        datum = component.get_datum(datum_id)
        
        return self.transformer.build_transformation_chain(datum, self)
    
    def transform_datum_to_global(self, component_name: str, datum_id: str, 
                                 point: List[float] = [0, 0, 0]) -> Tuple[np.ndarray, Dict]:
        """
        Transform a point from a datum's coordinate system to global coordinates.
        
        Args:
            component_name: Name of the component
            datum_id: ID of the datum
            point: Point in datum coordinates [x, y, z]
            
        Returns:
            Tuple of (transformed_point, transformation_info)
        """
        transformation_chain = self.get_datum_global_transform(component_name, datum_id)
        
        if not transformation_chain:
            # No transformation needed - already in global coordinates
            return np.array(point + [1]), {'steps': 0, 'chain': []}
        
        # Compose transformations
        composed_matrix, parameters, tolerances = self.transformer.compose_transformations(transformation_chain)
        
        # Create homogeneous coordinate vector
        point_homogeneous = [point[0], point[1], point[2], 1]
        
        # Evaluate the symbolic matrix numerically
        import sympy as sp
        eval_dict = {sp.symbols(k): v for k, v in parameters.items()}
        matrix_numeric = composed_matrix.subs(eval_dict)
        
        # Convert to numpy and apply transformation
        matrix_np = np.array(matrix_numeric.tolist(), dtype=float)
        point_np = np.array(point_homogeneous)
        transformed = matrix_np @ point_np
        
        transformation_info = {
            'steps': len(transformation_chain),
            'chain': [step.description for step in transformation_chain],
            'parameters': parameters,
            'tolerances': tolerances
        }
        
        return transformed[:3], transformation_info  # Return only x,y,z (not homogeneous)
    
    def transform_between_datums(self, source_comp: str, source_datum: str,
                               target_comp: str, target_datum: str,
                               point: List[float] = [0, 0, 0]) -> Tuple[np.ndarray, Dict]:
        """
        Transform a point from one datum's coordinate system to another datum's coordinate system.
        
        Args:
            source_comp: Source component name
            source_datum: Source datum ID
            target_comp: Target component name  
            target_datum: Target datum ID
            point: Point in source datum coordinates
            
        Returns:
            Tuple of (transformed_point, transformation_info)
        """
        # Transform source point to global coordinates
        global_point, source_info = self.transform_datum_to_global(source_comp, source_datum, point)
        
        # Get transformation from target datum to global
        target_chain = self.get_datum_global_transform(target_comp, target_datum)
        
        if not target_chain:
            # Target is already in global coordinates
            return global_point, {
                'source_steps': source_info['steps'],
                'target_steps': 0,
                'total_steps': source_info['steps']
            }
        
        # Compose target transformation and invert it
        target_matrix, target_params, target_tolerances = self.transformer.compose_transformations(target_chain)
        
        # Evaluate target matrix numerically
        import sympy as sp
        eval_dict = {sp.symbols(k): v for k, v in target_params.items()}
        target_matrix_numeric = target_matrix.subs(eval_dict)
        target_matrix_np = np.array(target_matrix_numeric.tolist(), dtype=float)
        
        # Invert the target transformation
        target_matrix_inv = np.linalg.inv(target_matrix_np)
        
        # Apply inverse transformation to get point in target coordinate system
        global_homogeneous = np.array([global_point[0], global_point[1], global_point[2], 1])
        target_point = target_matrix_inv @ global_homogeneous
        
        transformation_info = {
            'source_steps': source_info['steps'],
            'target_steps': len(target_chain),
            'total_steps': source_info['steps'] + len(target_chain),
            'source_chain': source_info['chain'],
            'target_chain': [step.description for step in target_chain]
        }
        
        return target_point[:3], transformation_info
    
    def analyze_tolerance_stackup(self, component_name: str, datum_id: str,
                                point: List[float] = [0, 0, 0]) -> Dict:
        """
        Analyze tolerance stackup for a point in a datum's coordinate system.
        
        Args:
            component_name: Name of the component
            datum_id: ID of the datum
            point: Point to analyze [x, y, z]
            
        Returns:
            Dictionary with nominal position and uncertainty estimates
        """
        # Get transformation chain
        transformation_chain = self.get_datum_global_transform(component_name, datum_id)
        
        if not transformation_chain:
            return {
                'nominal_position': point,
                'global_position': point,
                'uncertainty': {'std_x': 0.0, 'std_y': 0.0, 'std_z': 0.0},
                'transformation_steps': 0,
                'toleranced_dimensions': 0
            }
        
        # Get nominal global position
        global_point, transform_info = self.transform_datum_to_global(component_name, datum_id, point)
        
        # Propagate errors through transformation chain
        error_analysis = self.transformer.propagate_errors(transformation_chain, point)
        
        # Count toleranced dimensions
        toleranced_dims = sum(1 for step in transformation_chain if step.tolerances)
        
        return {
            'nominal_position': point,
            'global_position': global_point.tolist(),
            'uncertainty': error_analysis,
            'transformation_steps': len(transformation_chain),
            'toleranced_dimensions': toleranced_dims,
            'transformation_chain': [step.description for step in transformation_chain],
            'parameters': transform_info['parameters'],
            'tolerances': transform_info['tolerances']
        }
    
    def get_critical_dimensions(self) -> List[Dict]:
        """
        Identify critical dimensions that contribute most to positional uncertainty.
        
        Returns:
            List of dictionaries describing critical dimensions and their contributions
        """
        critical_dims = []
        
        # Analyze each component's datums
        for comp_name, component in self.components.items():
            if component == self.ground_component:
                continue  # Skip ground component
            
            for datum_id, datum in component.datums.items():
                # Analyze tolerance stackup for this datum
                analysis = self.analyze_tolerance_stackup(comp_name, datum_id)
                
                if analysis['toleranced_dimensions'] > 0:
                    # Calculate total positional uncertainty
                    uncertainty = analysis['uncertainty']
                    total_std = (uncertainty.get('std_x', 0)**2 + 
                               uncertainty.get('std_y', 0)**2 + 
                               uncertainty.get('std_z', 0)**2)**0.5
                    
                    critical_dims.append({
                        'component': comp_name,
                        'datum': datum_id,
                        'datum_name': datum.name,
                        'total_uncertainty': total_std,
                        'uncertainty_breakdown': uncertainty,
                        'contributing_tolerances': analysis['tolerances'],
                        'transformation_steps': analysis['transformation_steps']
                    })
        
        # Sort by total uncertainty (highest first)
        critical_dims.sort(key=lambda x: x['total_uncertainty'], reverse=True)
        
        return critical_dims
    
    def print_transformation_tree(self, component_name: str, datum_id: str):
        """
        Print a human-readable transformation tree for a datum.
        """
        print(f"\n=== Transformation Tree: {component_name}.{datum_id} ===")
        
        transformation_chain = self.get_datum_global_transform(component_name, datum_id)
        
        if not transformation_chain:
            print("  └─ No transformations (already in global coordinates)")
            return
        
        print(f"  Chain length: {len(transformation_chain)} steps")
        
        for i, step in enumerate(transformation_chain):
            indent = "  " + "│ " * i + "├─ " if i < len(transformation_chain) - 1 else "  " + "│ " * i + "└─ "
            print(f"{indent}{step.description}")
            
            if step.parameters:
                param_indent = "  " + "│ " * (i + 1)
                for param, value in step.parameters.items():
                    tol_info = ""
                    if param in step.tolerances:
                        tol_info = f" ±{step.tolerances[param]:.3f}"
                    print(f"{param_indent}  {param} = {value:.3f}{tol_info}")
        
        print(f"  └─ GLOBAL COORDINATES")
    
    def export_transformation_matrix(self, component_name: str, datum_id: str) -> str:
        """
        Export the symbolic transformation matrix as a string for external use.
        
        Returns:
            String representation of the symbolic transformation matrix
        """
        transformation_chain = self.get_datum_global_transform(component_name, datum_id)
        
        if not transformation_chain:
            return "Identity Matrix (no transformation needed)"
        
        composed_matrix, parameters, tolerances = self.transformer.compose_transformations(transformation_chain)
        
        matrix_str = "Symbolic Transformation Matrix:\n"
        matrix_str += str(composed_matrix) + "\n\n"
        
        matrix_str += "Parameters:\n"
        for param, value in parameters.items():
            tol_info = f" ±{tolerances[param]}" if param in tolerances else ""
            matrix_str += f"  {param} = {value}{tol_info}\n"
        
        return matrix_str