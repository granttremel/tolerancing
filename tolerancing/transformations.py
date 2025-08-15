"""
Coordinate transformation and error propagation system for tolerance analysis.

This module provides symbolic transformation matrices and error propagation
for navigating datum transfer trees in assemblies.
"""

from typing import List, Dict, Tuple, Optional, Union, TYPE_CHECKING
import numpy as np
import sympy as sp
from sympy import Matrix, symbols, cos, sin, sqrt, diff
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from .datum import Dimension, Datum
    from .assembly import Mate, Assembly

# from .datum import Datum
from .geometry import GeometryType

# Symbolic variables for transformations
x, y, z = symbols('x y z', real=True)
dx, dy, dz = symbols('dx dy dz', real=True)
rx, ry, rz = symbols('rx ry rz', real=True)  # rotation angles
t = symbols('t', real=True)  # parameter for parametric transformations

class TransformationType(Enum):
    """Types of coordinate transformations"""
    TRANSLATION = "translation"
    ROTATION = "rotation" 
    SCALING = "scaling"
    DIMENSION_OFFSET = "dimension_offset"
    MATE_CONSTRAINT = "mate_constraint"

@dataclass
class TransformationStep:
    """
    Represents a single step in a coordinate transformation chain.
    Each step has a symbolic transformation matrix and associated tolerances.
    """
    transform_type: TransformationType
    symbolic_matrix: Matrix
    parameters: Dict[str, Union[float, sp.Symbol]]
    tolerances: Dict[str, float]
    source_datum: Optional['Datum'] = None
    target_datum: Optional['Datum'] = None
    description: str = ""
    
    def evaluate(self, values: Dict[str, float]) -> np.ndarray:
        """Evaluate the symbolic matrix with numerical values"""
        matrix_func = sp.lambdify(list(self.parameters.keys()), self.symbolic_matrix, 'numpy')
        param_values = [values.get(param, self.parameters[param]) for param in self.parameters.keys()]
        return matrix_func(*param_values)
    
    def get_jacobian(self, variables: List[str]) -> Matrix:
        """Get the Jacobian matrix for error propagation"""
        jacobian_elements = []
        for i in range(self.symbolic_matrix.rows):
            row = []
            for var in variables:
                if var in self.parameters:
                    derivative = diff(self.symbolic_matrix[i], symbols(var))
                    row.append(derivative)
                else:
                    row.append(0)
            jacobian_elements.append(row)
        return Matrix(jacobian_elements)

class CoordinateTransformer:
    """
    Manages coordinate transformations and error propagation for datum transfer chains.
    """
    
    def __init__(self):
        self.cached_transforms = {}
        self.transformation_chains = {}
    
    @classmethod
    def rotate_3d(cls, frame, rx, ry, rz):
        
        sz = frame.shape
        if not sz== (4,4):
            newframe = np.eye(4)
            newframe[:sz[0], :sz[1]] = frame
        else:
            newframe = frame.copy()
            
        rot = cls.create_rotation_matrix_xyz(rx, ry, rz)
        rotframe = rot@newframe
        return rotframe[:sz[0], :sz[1]]
    
    @classmethod
    def create_translation_matrix(cls, dx: float|sp.Symbol, dy: float|sp.Symbol, dz: float|sp.Symbol) -> Matrix:
        """Create a symbolic 4x4 translation matrix"""
        return np.array([
            [1, 0, 0, dx],
            [0, 1, 0, dy], 
            [0, 0, 1, dz],
            [0, 0, 0, 1]
        ])
    
    @classmethod
    def create_rotation_matrix_z(cls, angle: float|sp.Symbol) -> np.array:
        """Create a symbolic 4x4 rotation matrix around Z axis"""
        return np.array([
            [cos(angle), -sin(angle), 0, 0],
            [sin(angle), cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    @classmethod
    def create_rotation_matrix_xyz(cls, rx: float|sp.Symbol, ry: float|sp.Symbol, rz: float|sp.Symbol) -> np.array:
        """Create a symbolic 4x4 rotation matrix for XYZ Euler angles"""
        # Rotation around X
        Rx = np.array([
            [1, 0, 0, 0],
            [0, cos(rx), -sin(rx), 0],
            [0, sin(rx), cos(rx), 0],
            [0, 0, 0, 1]
        ])
        
        # Rotation around Y  
        Ry = np.array([
            [cos(ry), 0, sin(ry), 0],
            [0, 1, 0, 0],
            [-sin(ry), 0, cos(ry), 0],
            [0, 0, 0, 1]
        ])
        
        # Rotation around Z
        Rz = np.array([
            [cos(rz), -sin(rz), 0, 0],
            [sin(rz), cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        return Rx * Ry * Rz
    
    @classmethod
    def create_sym_translation_matrix(cls):
        
        dx_sym = sp.Symbol('dx')
        dy_sym = sp.Symbol('dy')
        dz_sym = sp.Symbol('dz')
        return Matrix(cls.create_translation_matrix(dx_sym, dy_sym, dz_sym)), (dx_sym, dy_sym, dz_sym)
        
    @classmethod
    def create_sym_rotation_matrix_z(cls):
        
        rz_sym = sp.Symbol('rz')
        return Matrix(cls.create_rotation_matrix_z(rz_sym)), (rz_sym,)
    
    @classmethod
    def create_sym_rotation_matrix_xyz(cls):
        
        rx_sym = sp.Symbol('rx')
        ry_sym = sp.Symbol('ry')
        rz_sym = sp.Symbol('rz')
        return Matrix(cls.create_rotation_matrix_xyz(rx_sym, ry_sym, rz_sym)), (rx_sym, ry_sym, rz_sym)

    def create_dimension_transform(self, dimension: 'Dimension') -> TransformationStep:
        """
        Create a transformation step from a datum dimension.
        """
        if dimension.dimension_type == 'offset_distance':
            # Translation by distance in specified direction
            direction = np.array(dimension.constraints.get('direction', [0, 0, 1]))
            distance_sym = symbols(f'd_{dimension.id}')
            
            dx_sym = distance_sym * direction[0]
            dy_sym = distance_sym * direction[1] 
            dz_sym = distance_sym * direction[2]
            
            matrix = self.create_translation_matrix(dx_sym, dy_sym, dz_sym)
            
            parameters = {f'd_{dimension.id}': dimension.values[0] if hasattr(dimension.values, '__len__') else dimension.values}
            tolerances = {}
            
            if dimension.tolerance:
                tolerances[f'd_{dimension.id}'] = dimension.tolerance.pos
            
            return TransformationStep(
                transform_type=TransformationType.DIMENSION_OFFSET,
                symbolic_matrix=matrix,
                parameters=parameters,
                tolerances=tolerances,
                description=f"Offset distance {dimension.dimension_type}"
            )
            
        elif dimension.dimension_type == 'offset_xy':
            # Translation by X,Y offsets
            dx_sym = symbols(f'dx_{dimension.id}')
            dy_sym = symbols(f'dy_{dimension.id}')
            dz_sym = 0
            
            matrix = self.create_translation_matrix(dx_sym, dy_sym, dz_sym)
            
            parameters = {
                f'dx_{dimension.id}': dimension.values[0],
                f'dy_{dimension.id}': dimension.values[1]
            }
            tolerances = {}
            
            if dimension.tolerance:
                tolerances[f'dx_{dimension.id}'] = dimension.tolerance.pos
                tolerances[f'dy_{dimension.id}'] = dimension.tolerance.pos
            
            return TransformationStep(
                transform_type=TransformationType.DIMENSION_OFFSET,
                symbolic_matrix=matrix,
                parameters=parameters,
                tolerances=tolerances,
                description=f"Offset XY {dimension.dimension_type}"
            )
            
        elif dimension.dimension_type == 'angle_from_axis':
            # Rotation by angle
            angle_sym = symbols(f'a_{dimension.id}')
            matrix = self.create_rotation_matrix_z(angle_sym)
            
            parameters = {f'a_{dimension.id}': dimension.values[0] if hasattr(dimension.values, '__len__') else dimension.values}
            tolerances = {}
            
            if dimension.tolerance:
                tolerances[f'a_{dimension.id}'] = dimension.tolerance.pos
            
            return TransformationStep(
                transform_type=TransformationType.ROTATION,
                symbolic_matrix=matrix,
                parameters=parameters,
                tolerances=tolerances,
                description=f"Rotation {dimension.dimension_type}"
            )
        
        else:
            # Default identity transform for unsupported dimension types
            matrix = Matrix.eye(4)
            return TransformationStep(
                transform_type=TransformationType.TRANSLATION,
                symbolic_matrix=matrix,
                parameters={},
                tolerances={},
                description=f"Identity for {dimension.dimension_type}"
            )
    
    def create_mate_transform(self, mate: 'Mate') -> TransformationStep:
        """
        Create a transformation step from an assembly mate constraint.
        """
        from .assembly import MateType
        
        if mate.mate_type == MateType.COINCIDENT:
            # COINCIDENT mate: Proper DOF constraint based on geometry types
            datum1, datum2 = mate.get_datums()
            geo1_type = datum1.geo.geotype
            geo2_type = datum2.geo.geotype
            
            from .geometry import GeometryType
            
            # Calculate offset and orientation alignment
            offset = datum1.geo.aorigin - datum2.geo.aorigin
            
            if geo1_type == GeometryType.PLANE and geo2_type == GeometryType.PLANE:
                # PLANE-PLANE COINCIDENT: 
                # - Constrains 1 translation DOF (along normal)
                # - Constrains 2 rotation DOFs (normal alignment)
                # - Leaves 2 translation DOFs free (in-plane motion)
                # - Leaves 1 rotation DOF free (rotation about normal)
                
                # Align normals (rotation constraint)
                normal1 = datum1.geo.aframe
                normal2 = datum2.geo.aframe
                
                # Calculate rotation to align normals
                cross_product = np.cross(normal2, normal1)
                dot_product = np.dot(normal2, normal1)
                
                if np.linalg.norm(cross_product) > 1e-10:  # Not parallel
                    # Rotation axis and angle
                    axis = cross_product / np.linalg.norm(cross_product)
                    angle = np.arccos(np.clip(dot_product, -1, 1))
                    
                    # Create symbolic rotation matrix
                    rx_sym = symbols(f'mate_rx_{mate.id}')
                    ry_sym = symbols(f'mate_ry_{mate.id}')
                    rz_sym = symbols(f'mate_rz_{mate.id}')
                    rotation_matrix = self.create_rotation_matrix_xyz(rx_sym, ry_sym, rz_sym)
                    
                    # Distance along normal to align planes
                    normal_distance = np.dot(offset, normal1)
                    dz_sym = symbols(f'mate_dz_{mate.id}')
                    translation_matrix = self.create_translation_matrix(0, 0, dz_sym)
                    
                    matrix = translation_matrix * rotation_matrix
                    
                    parameters = {
                        f'mate_rx_{mate.id}': angle * axis[0] if np.linalg.norm(axis) > 0 else 0,
                        f'mate_ry_{mate.id}': angle * axis[1] if np.linalg.norm(axis) > 0 else 0,
                        f'mate_rz_{mate.id}': angle * axis[2] if np.linalg.norm(axis) > 0 else 0,
                        f'mate_dz_{mate.id}': normal_distance
                    }
                else:
                    # Planes already parallel, just translation
                    normal_distance = np.dot(offset, normal1)
                    dz_sym = symbols(f'mate_dz_{mate.id}')
                    matrix = self.create_translation_matrix(0, 0, dz_sym)
                    parameters = {f'mate_dz_{mate.id}': normal_distance}
                    
            elif geo1_type == GeometryType.AXIS and geo2_type == GeometryType.AXIS:
                # AXIS-AXIS COINCIDENT:
                # - Constrains 2 translation DOFs (perpendicular to axes)
                # - Constrains 2 rotation DOFs (axis alignment)
                # - Leaves 1 translation DOF free (along axis)
                # - Leaves 1 rotation DOF free (clocking about axis)
                
                axis1 = datum1.geo.aframe
                axis2 = datum2.geo.aframe
                
                # Align axes (rotation constraint)
                cross_product = np.cross(axis2, axis1)
                dot_product = np.dot(axis2, axis1)
                
                if np.linalg.norm(cross_product) > 1e-10:  # Not parallel
                    axis = cross_product / np.linalg.norm(cross_product)
                    angle = np.arccos(np.clip(dot_product, -1, 1))
                    
                    rx_sym = symbols(f'mate_rx_{mate.id}')
                    ry_sym = symbols(f'mate_ry_{mate.id}')
                    rz_sym = symbols(f'mate_rz_{mate.id}')
                    rotation_matrix = self.create_rotation_matrix_xyz(rx_sym, ry_sym, rz_sym)
                    
                    # Translation to align axis origins (perpendicular components only)
                    axis_offset = offset - np.dot(offset, axis1) * axis1  # Remove parallel component
                    dx_sym = symbols(f'mate_dx_{mate.id}')
                    dy_sym = symbols(f'mate_dy_{mate.id}')
                    translation_matrix = self.create_translation_matrix(dx_sym, dy_sym, 0)
                    
                    matrix = translation_matrix * rotation_matrix
                    
                    parameters = {
                        f'mate_rx_{mate.id}': angle * axis[0] if np.linalg.norm(axis) > 0 else 0,
                        f'mate_ry_{mate.id}': angle * axis[1] if np.linalg.norm(axis) > 0 else 0, 
                        f'mate_rz_{mate.id}': angle * axis[2] if np.linalg.norm(axis) > 0 else 0,
                        f'mate_dx_{mate.id}': axis_offset[0],
                        f'mate_dy_{mate.id}': axis_offset[1]
                    }
                else:
                    # Axes already parallel, just perpendicular translation
                    axis_offset = offset - np.dot(offset, axis1) * axis1
                    dx_sym = symbols(f'mate_dx_{mate.id}')
                    dy_sym = symbols(f'mate_dy_{mate.id}')
                    matrix = self.create_translation_matrix(dx_sym, dy_sym, 0)
                    parameters = {
                        f'mate_dx_{mate.id}': axis_offset[0],
                        f'mate_dy_{mate.id}': axis_offset[1]
                    }
                    
            else:
                # POINT-POINT or MIXED GEOMETRY COINCIDENT:
                # - Constrains all 3 translation DOFs
                # - Rotation constraints depend on geometry types
                dx_sym = symbols(f'mate_dx_{mate.id}')
                dy_sym = symbols(f'mate_dy_{mate.id}')
                dz_sym = symbols(f'mate_dz_{mate.id}')
                
                matrix = self.create_translation_matrix(dx_sym, dy_sym, dz_sym)
                
                parameters = {
                    f'mate_dx_{mate.id}': offset[0],
                    f'mate_dy_{mate.id}': offset[1],
                    f'mate_dz_{mate.id}': offset[2]
                }
            
            return TransformationStep(
                transform_type=TransformationType.MATE_CONSTRAINT,
                symbolic_matrix=matrix,
                parameters=parameters,
                tolerances={},  # Mates are constraints, not toleranced dimensions
                description=f"Mate constraint: {mate.mate_type.value} ({geo1_type.name}-{geo2_type.name})"
            )
            
        elif mate.mate_type == MateType.CONCENTRIC:
            # CONCENTRIC mate: Align axes centers, allow rotation about axis
            datum1, datum2 = mate.get_datums()
            geo1_type = datum1.geo.geotype
            geo2_type = datum2.geo.geotype
            
            from .geometry import GeometryType
            
            # For concentric mates (cylinders, axes), constrain radial position only
            offset = datum1.geo.aorigin - datum2.geo.aorigin
            
            if (geo1_type in [GeometryType.AXIS, GeometryType.CYLINDER] and 
                geo2_type in [GeometryType.AXIS, GeometryType.CYLINDER]):
                
                # Align axes centers (radial constraint only)
                axis1 = datum1.geo.aframe
                axis2 = datum2.geo.aframe
                
                # Project offset perpendicular to axes (radial alignment)
                radial_offset = offset - np.dot(offset, axis1) * axis1
                
                dx_sym = symbols(f'mate_dx_{mate.id}')
                dy_sym = symbols(f'mate_dy_{mate.id}')
                
                matrix = self.create_translation_matrix(dx_sym, dy_sym, 0)
                
                parameters = {
                    f'mate_dx_{mate.id}': radial_offset[0],
                    f'mate_dy_{mate.id}': radial_offset[1]
                }
            else:
                # Fallback for other geometry combinations
                dx_sym = symbols(f'mate_dx_{mate.id}')
                dy_sym = symbols(f'mate_dy_{mate.id}')
                dz_sym = symbols(f'mate_dz_{mate.id}')
                
                matrix = self.create_translation_matrix(dx_sym, dy_sym, dz_sym)
                
                parameters = {
                    f'mate_dx_{mate.id}': offset[0],
                    f'mate_dy_{mate.id}': offset[1], 
                    f'mate_dz_{mate.id}': offset[2]
                }
            
            return TransformationStep(
                transform_type=TransformationType.MATE_CONSTRAINT,
                symbolic_matrix=matrix,
                parameters=parameters,
                tolerances={},
                description=f"Mate constraint: {mate.mate_type.value} ({geo1_type.name}-{geo2_type.name})"
            )
        
        else:
            # Default identity for other mate types
            matrix = Matrix.eye(4)
            return TransformationStep(
                transform_type=TransformationType.MATE_CONSTRAINT,
                symbolic_matrix=matrix,
                parameters={},
                tolerances={},
                description=f"Identity mate: {mate.mate_type.value}"
            )
    
    def build_transformation_chain(self, datum: 'Datum', assembly: 'Assembly') -> List[TransformationStep]:
        """
        Build the complete transformation chain from a datum to the global coordinate system.
        
        Args:
            datum: Target datum to transform
            assembly: Assembly containing the datum
            
        Returns:
            List of transformation steps from datum to global coordinates
        """
        chain = []
        current_datum = datum
        
        # First, follow the datum construction chain within the component
        construction_chain = datum.get_construction_chain()
        
        for i in range(len(construction_chain) - 1, 0, -1):
            derived_datum = construction_chain[i]
            if derived_datum.dimension:
                transform_step = self.create_dimension_transform(derived_datum.dimension)
                transform_step.source_datum = construction_chain[i-1]
                transform_step.target_datum = derived_datum
                chain.append(transform_step)
        
        # Then, follow mate relationships to other components
        component = datum.parent
        if component and component.parent:  # If component is in an assembly
            # Find mates involving this component
            for mate in assembly.mates:
                if mate.component1 == component:
                    # This component is mated to another
                    mate_transform = self.create_mate_transform(mate)
                    mate_transform.source_datum = component.get_datum(mate.datum1_id)
                    mate_transform.target_datum = mate.component2.get_datum(mate.datum2_id)
                    chain.append(mate_transform)
                    
                    # Continue chain through the mated component if it's not ground
                    if not mate.component2.is_positioned or mate.component2 != assembly.ground_component:
                        # Recursively build chain for mated component
                        mate_chain = self.build_transformation_chain(mate.component2.get_datum(mate.datum2_id), assembly)
                        chain.extend(mate_chain)
                    break
                    
                elif mate.component2 == component:
                    # This component is the target of a mate
                    mate_transform = self.create_mate_transform(mate)
                    mate_transform.source_datum = mate.component1.get_datum(mate.datum1_id) 
                    mate_transform.target_datum = component.get_datum(mate.datum2_id)
                    chain.append(mate_transform)
                    
                    # Continue chain through the mating component if it's not ground
                    if not mate.component1.is_positioned or mate.component1 != assembly.ground_component:
                        mate_chain = self.build_transformation_chain(mate.component1.get_datum(mate.datum1_id), assembly)
                        chain.extend(mate_chain)
                    break
        
        return chain
    
    def compose_transformations(self, transformation_chain: List[TransformationStep]) -> Tuple[Matrix, Dict, Dict]:
        """
        Compose a list of transformation steps into a single transformation matrix.
        
        Returns:
            Tuple of (composed_matrix, combined_parameters, combined_tolerances)
        """
        if not transformation_chain:
            return Matrix.eye(4), {}, {}
        
        # Start with identity
        composed_matrix = Matrix.eye(4)
        combined_parameters = {}
        combined_tolerances = {}
        
        # Compose transformations from right to left (last applied first)
        for step in reversed(transformation_chain):
            composed_matrix = step.symbolic_matrix * composed_matrix
            combined_parameters.update(step.parameters)
            combined_tolerances.update(step.tolerances)
        
        return composed_matrix, combined_parameters, combined_tolerances
    
    def propagate_errors(self, transformation_chain: List[TransformationStep], 
                        point: List[float] = [0, 0, 0]) -> Dict[str, float]:
        """
        Use error propagation to estimate variance at a point due to tolerances.
        
        Args:
            transformation_chain: Chain of transformations
            point: Point to evaluate variance at [x, y, z]
            
        Returns:
            Dictionary with variance estimates for x, y, z coordinates
        """
        # Compose the full transformation
        composed_matrix, parameters, tolerances = self.compose_transformations(transformation_chain)
        
        if not tolerances:
            return {'var_x': 0.0, 'var_y': 0.0, 'var_z': 0.0}
        
        # Create homogeneous coordinate vector
        point_vector = Matrix([point[0], point[1], point[2], 1])
        
        # Apply transformation
        transformed_point = composed_matrix * point_vector
        
        # Calculate Jacobian matrix (partial derivatives)
        tolerance_vars = [symbols(var) for var in tolerances.keys()]
        jacobian_elements = []
        
        for i in range(3):  # Only x, y, z components (not homogeneous coordinate)
            row = []
            for var in tolerance_vars:
                derivative = diff(transformed_point[i], var)
                row.append(derivative)
            jacobian_elements.append(row)
        
        jacobian = Matrix(jacobian_elements)
        
        # Evaluate Jacobian at nominal parameter values
        eval_dict = {symbols(k): v for k, v in parameters.items()}
        jacobian_numeric = jacobian.subs(eval_dict)
        
        # Create covariance matrix for input tolerances (assuming independent)
        tolerance_vars_numeric = [tolerances[str(var)] for var in tolerance_vars]
        input_covariance = Matrix.diag(*[tol**2 for tol in tolerance_vars_numeric])
        
        # Propagate covariance: Cov_out = J * Cov_in * J^T
        try:
            jacobian_float = Matrix([[float(jacobian_numeric[i, j]) for j in range(jacobian_numeric.cols)] 
                                   for i in range(jacobian_numeric.rows)])
            
            output_covariance = jacobian_float * input_covariance * jacobian_float.T
            
            return {
                'var_x': float(output_covariance[0, 0]),
                'var_y': float(output_covariance[1, 1]),
                'var_z': float(output_covariance[2, 2]),
                'std_x': float(sqrt(output_covariance[0, 0])),
                'std_y': float(sqrt(output_covariance[1, 1])),
                'std_z': float(sqrt(output_covariance[2, 2]))
            }
        except Exception as e:
            print(f"Error in variance calculation: {e}")
            return {'var_x': 0.0, 'var_y': 0.0, 'var_z': 0.0}