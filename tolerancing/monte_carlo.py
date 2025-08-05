"""
Monte Carlo tolerance analysis system for assembly simulation.

This module provides Monte Carlo sampling of toleranced dimensions and 
measurement calculation for statistical analysis of assembly behavior.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import csv
from pathlib import Path

from .assembly import Assembly
from .datum import Datum
from .component import Component
from .transformations import CoordinateTransformer

class MeasurementType(Enum):
    """Types of measurements between datums"""
    DISTANCE = "distance"                    # Distance between origins
    POSITION_3D = "position_3d"             # 3D position difference (X, Y, Z)
    ORIENTATION_3D = "orientation_3d"       # 3D orientation difference (RX, RY, RZ)
    FULL_6DOF = "full_6dof"                 # Complete 6-DOF relationship
    PARALLELISM = "parallelism"             # Angular deviation from parallel
    PERPENDICULARITY = "perpendicularity"   # Angular deviation from perpendicular
    CONCENTRICITY = "concentricity"         # Radial deviation from concentric

@dataclass
class Measurement:
    """
    Defines a measurement between two datums in an assembly.
    Measurements quantify geometric relationships for tolerance analysis.
    """
    name: str
    measurement_type: MeasurementType
    source_component: str
    source_datum: str
    target_component: str
    target_datum: str
    description: str = ""
    
    def __post_init__(self):
        if not self.description:
            self.description = f"{self.measurement_type.value} between {self.source_component}.{self.source_datum} and {self.target_component}.{self.target_datum}"
            
    def __str__(self):
        
        return f""

class MonteCarloAnalyzer:
    """
    Performs Monte Carlo tolerance analysis on assemblies.
    
    Samples toleranced dimensions according to normal distributions,
    instantiates assemblies, and calculates measurements for statistical analysis.
    """
    
    def __init__(self, assembly: Assembly):
        self.assembly = assembly
        self.measurements: List[Measurement] = []
        self.results_history: List[Dict] = []
        
        # Extract all toleranced dimensions from the assembly
        self.toleranced_dimensions = self._extract_toleranced_dimensions()
        
    def _extract_toleranced_dimensions(self) -> List[Dict]:
        """Extract all toleranced dimensions from the assembly"""
        dimensions = []
        
        for comp_name, component in self.assembly.components.items():
            for datum_id, datum in component.datums.items():
                if datum.dimension and datum.dimension.tolerance:
                    # Dimensional tolerances (from parametric datums)
                    dim_info = {
                        'type': 'dimension',
                        'component': comp_name,
                        'datum': datum_id,
                        'datum_name': datum.name,
                        'dimension_id': datum.dimension.id,
                        'dimension_name': datum.dimension.name,
                        'dimension_type': datum.dimension.dimension_type,
                        'nominal_values': datum.dimension.values,
                        'tolerance': datum.dimension.tolerance,
                        'parameter_names': [f'd_{datum.dimension.id}'] if datum.dimension.dimension_type == 'offset_distance' 
                                         else [f'dx_{datum.dimension.id}', f'dy_{datum.dimension.id}'] if datum.dimension.dimension_type == 'offset_xy'
                                         else [f'a_{datum.dimension.id}'] if datum.dimension.dimension_type == 'angle_from_axis'
                                         else []
                    }
                    dimensions.append(dim_info)
                
                elif datum.tol:
                    # Geometric tolerances (position, form, etc.)
                    tol_info = {
                        'type': 'geometric',
                        'component': comp_name,
                        'datum': datum_id,
                        'datum_name': datum.name,
                        'tolerance_type': datum.tol.toltype,
                        'tolerance_value': datum.tol.pos,
                        'tolerance': datum.tol
                    }
                    dimensions.append(tol_info)
        
        return dimensions
    
    def add_measurement(self, measurement: Measurement):
        """Add a measurement to be calculated during Monte Carlo analysis"""
        # Validate that the datums exist
        if measurement.source_component not in self.assembly.components:
            raise ValueError(f"Source component '{measurement.source_component}' not found")
        if measurement.target_component not in self.assembly.components:
            raise ValueError(f"Target component '{measurement.target_component}' not found")
        
        source_comp = self.assembly.components[measurement.source_component]
        target_comp = self.assembly.components[measurement.target_component]
        
        if measurement.source_datum not in source_comp.datums:
            raise ValueError(f"Source datum '{measurement.source_datum}' not found in {measurement.source_component}")
        if measurement.target_datum not in target_comp.datums:
            raise ValueError(f"Target datum '{measurement.target_datum}' not found in {measurement.target_component}")
        
        self.measurements.append(measurement)
    
    def sample_dimensions(self, n_samples: int = 1000, sigma_factor: float = 3.0) -> List[Dict[str, float]]:
        """
        Generate Monte Carlo samples of all toleranced dimensions.
        
        Args:
            n_samples: Number of Monte Carlo samples to generate
            sigma_factor: Factor relating tolerance limits to standard deviation (default: 3.0 for ±3σ)
            
        Returns:
            List of dictionaries, each containing sampled values for all parameters
        """
        samples = []
        
        for i in range(n_samples):
            sample = {}
            
            for dim_info in self.toleranced_dimensions:
                if dim_info['type'] == 'dimension':
                    # Sample dimensional tolerances
                    tolerance = dim_info['tolerance']
                    nominal_values = dim_info['nominal_values']
                    param_names = dim_info['parameter_names']
                    
                    # Convert tolerance limits to standard deviation
                    # Assuming symmetric tolerance: ±tol corresponds to ±3σ
                    pos_sigma = tolerance.pos / sigma_factor
                    neg_sigma = tolerance.neg / sigma_factor if tolerance.neg else pos_sigma
                    
                    if isinstance(nominal_values, (list, np.ndarray)):
                        # Multiple values (e.g., offset_xy)
                        for j, (param_name, nominal) in enumerate(zip(param_names, nominal_values)):
                            # Use average of positive and negative sigma for simplicity
                            sigma = (pos_sigma + neg_sigma) / 2
                            sampled_value = np.random.normal(nominal, sigma)
                            sample[param_name] = sampled_value
                    else:
                        # Single value (e.g., offset_distance)
                        if param_names:
                            sigma = (pos_sigma + neg_sigma) / 2
                            sampled_value = np.random.normal(nominal_values, sigma)
                            sample[param_names[0]] = sampled_value
                
                elif dim_info['type'] == 'geometric':
                    # Sample geometric tolerances (position, flatness, etc.)
                    tolerance_value = dim_info['tolerance_value']
                    tolerance_type = dim_info['tolerance_type']
                    sigma = tolerance_value / sigma_factor
                    
                    param_base = f"{tolerance_type.name.lower()}_{dim_info['component']}_{dim_info['datum']}"
                    
                    if tolerance_type.name == 'POSITION':
                        # Generate random position within circular tolerance zone
                        angle = np.random.uniform(0, 2*np.pi)
                        radius = np.random.normal(0, sigma)  # Normal distribution within zone
                        
                        sample[f"{param_base}_x"] = radius * np.cos(angle)
                        sample[f"{param_base}_y"] = radius * np.sin(angle)
                        sample[f"{param_base}_z"] = np.random.normal(0, sigma/3)  # Smaller Z variation
                        
                    elif tolerance_type.name == 'FLATNESS':
                        # Sample angular deviations for flatness (small angle approximation)
                        # Flatness causes the plane normal to deviate by small angles
                        # Convert flatness tolerance to angular deviation (simplified)
                        # For optical surfaces, flatness tolerance relates to surface irregularity
                        # Approximate: flatness error ≈ length × sin(angle) ≈ length × angle (small angles)
                        # Assume characteristic length of 10mm for conversion
                        characteristic_length = 10.0  # mm
                        max_angle_rad = tolerance_value / characteristic_length
                        angle_sigma = max_angle_rad / sigma_factor
                        
                        # Sample two orthogonal angular deviations (rx, ry)
                        sample[f"{param_base}_rx"] = np.random.normal(0, angle_sigma)
                        sample[f"{param_base}_ry"] = np.random.normal(0, angle_sigma)
                        
                    else:
                        # For other tolerance types, default to positional variation
                        angle = np.random.uniform(0, 2*np.pi)
                        radius = np.random.normal(0, sigma)
                        
                        sample[f"{param_base}_x"] = radius * np.cos(angle)
                        sample[f"{param_base}_y"] = radius * np.sin(angle)
                        sample[f"{param_base}_z"] = np.random.normal(0, sigma/3)
            
            samples.append(sample)
        
        return samples
    
    def _create_measurement_function(self, measurement: Measurement):
        """
        Create a lambda function that can calculate a measurement given parameter values.
        This approach is much more efficient and avoids modifying shared state.
        
        Returns:
            Function that takes parameter_sample dict and returns measurement results
        """
        # Get the transformation chains once
        source_chain = self.assembly.get_datum_global_transform(
            measurement.source_component, measurement.source_datum
        )
        target_chain = self.assembly.get_datum_global_transform(
            measurement.target_component, measurement.target_datum
        )
        
        # Compose the symbolic transformation matrices
        source_matrix, source_nominal_params, _ = self.assembly.transformer.compose_transformations(source_chain)
        target_matrix, target_nominal_params, _ = self.assembly.transformer.compose_transformations(target_chain)
        
        # Get datum orientations for orientation-based measurements
        source_datum = self.assembly.components[measurement.source_component].get_datum(measurement.source_datum)
        target_datum = self.assembly.components[measurement.target_component].get_datum(measurement.target_datum)
        source_orientation = source_datum.geo.aframe
        target_orientation = target_datum.geo.aframe
        
        def measurement_function(parameter_sample: Dict[str, float]) -> Dict[str, float]:
            """Calculate measurement for given parameter sample"""
            import sympy as sp
            
            # Merge nominal parameters with sampled values
            source_params = source_nominal_params.copy()
            target_params = target_nominal_params.copy()
            
            # Update with sampled values
            for param_name, sampled_value in parameter_sample.items():
                if param_name in source_params:
                    source_params[param_name] = sampled_value
                if param_name in target_params:
                    target_params[param_name] = sampled_value
            
            # Handle flatness angular deviations by modifying transformation matrices
            source_matrix_modified = source_matrix
            target_matrix_modified = target_matrix
            
            # Apply flatness rotations if present in the sample
            for param_name, sampled_value in parameter_sample.items():
                if 'flatness_' in param_name and '_rx' in param_name:
                    # Extract component and datum from parameter name
                    # Format: flatness_COMPONENT_DATUM_rx
                    parts = param_name.split('_')
                    if len(parts) >= 4:
                        component = parts[1]
                        datum = parts[2]
                        
                        # Find corresponding ry parameter
                        ry_param = param_name.replace('_rx', '_ry')
                        ry_value = parameter_sample.get(ry_param, 0.0)
                        
                        # Create small angle rotation matrix for flatness
                        rx_val = sampled_value
                        ry_val = ry_value
                        
                        # Small angle approximation: sin(θ) ≈ θ, cos(θ) ≈ 1
                        flatness_rotation = sp.Matrix([
                            [1, -ry_val, rx_val, 0],
                            [ry_val, 1, -rx_val, 0], 
                            [-rx_val, rx_val, 1, 0],
                            [0, 0, 0, 1]
                        ])
                        
                        # Apply rotation to appropriate transformation matrix
                        if measurement.source_component == component:
                            source_matrix_modified = flatness_rotation * source_matrix_modified
                        elif measurement.target_component == component: 
                            target_matrix_modified = flatness_rotation * target_matrix_modified
            
            # Evaluate source transformation
            if source_params:
                source_eval_dict = {sp.symbols(k): v for k, v in source_params.items()}
                source_matrix_numeric = source_matrix_modified.subs(source_eval_dict)
                source_matrix_np = np.array(source_matrix_numeric.tolist(), dtype=float)
                source_point = np.array([0, 0, 0, 1])
                source_global = (source_matrix_np @ source_point)[:3]
            else:
                source_global = np.array([0, 0, 0])
            
            # Evaluate target transformation
            if target_params:
                target_eval_dict = {sp.symbols(k): v for k, v in target_params.items()}
                target_matrix_numeric = target_matrix_modified.subs(target_eval_dict)
                target_matrix_np = np.array(target_matrix_numeric.tolist(), dtype=float)
                target_point = np.array([0, 0, 0, 1])
                target_global = (target_matrix_np @ target_point)[:3]
            else:
                target_global = np.array([0, 0, 0])
            
            # Calculate measurement based on type
            result = {}
            
            if measurement.measurement_type == MeasurementType.DISTANCE:
                distance = np.linalg.norm(target_global - source_global)
                result['distance'] = distance
                
            elif measurement.measurement_type == MeasurementType.POSITION_3D:
                delta = target_global - source_global
                result['delta_x'] = delta[0]
                result['delta_y'] = delta[1]
                result['delta_z'] = delta[2]
                
            elif measurement.measurement_type == MeasurementType.ORIENTATION_3D:
                # Calculate relative orientation (simplified)
                result['delta_rx'] = 0.0  # Placeholder
                result['delta_ry'] = 0.0  # Placeholder  
                result['delta_rz'] = 0.0  # Placeholder
                
            elif measurement.measurement_type == MeasurementType.FULL_6DOF:
                # Combine position and orientation
                delta = target_global - source_global
                result['delta_x'] = delta[0]
                result['delta_y'] = delta[1]
                result['delta_z'] = delta[2]
                result['delta_rx'] = 0.0  # Placeholder
                result['delta_ry'] = 0.0  # Placeholder
                result['delta_rz'] = 0.0  # Placeholder
                
            elif measurement.measurement_type == MeasurementType.PARALLELISM:
                # Angular deviation from parallel
                dot_product = np.dot(source_orientation, target_orientation)
                angle_rad = np.arccos(np.clip(abs(dot_product), 0, 1))
                result['parallelism_error'] = np.degrees(angle_rad)
                
            elif measurement.measurement_type == MeasurementType.PERPENDICULARITY:
                # Angular deviation from perpendicular
                dot_product = np.dot(source_orientation, target_orientation)
                angle_rad = np.arccos(np.clip(abs(dot_product), 0, 1))
                perp_error = abs(angle_rad - np.pi/2)
                result['perpendicularity_error'] = np.degrees(perp_error)
                
            elif measurement.measurement_type == MeasurementType.CONCENTRICITY:
                # Radial deviation from concentric (assuming both are axes/cylinders)
                radial_delta = target_global[:2] - source_global[:2]  # X, Y only
                radial_error = np.linalg.norm(radial_delta)
                result['concentricity_error'] = radial_error
            
            return result
        
        return measurement_function

    def calculate_measurement(self, measurement: Measurement, 
                            parameter_sample: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate a measurement value for a given parameter sample.
        This now uses the lambda function approach for efficiency.
        
        Args:
            measurement: Measurement definition
            parameter_sample: Dictionary of sampled parameter values
            
        Returns:
            Dictionary with measurement results
        """
        # Create or get cached measurement function
        measurement_key = f"{measurement.name}_{id(measurement)}"
        if not hasattr(self, '_measurement_functions'):
            self._measurement_functions = {}
        
        if measurement_key not in self._measurement_functions:
            self._measurement_functions[measurement_key] = self._create_measurement_function(measurement)
        
        # Use the cached function to calculate the measurement
        return self._measurement_functions[measurement_key](parameter_sample)
    
    def run_monte_carlo(self, n_samples: int = 1000, sigma_factor: float = 3.0) -> pd.DataFrame:
        """
        Run complete Monte Carlo analysis.
        
        Args:
            n_samples: Number of Monte Carlo samples
            sigma_factor: Factor relating tolerance limits to sigma
            
        Returns:
            DataFrame with all samples, dimensions, and measurements
        """
        print(f"Running Monte Carlo analysis with {n_samples} samples...")
        
        # Generate samples
        parameter_samples = self.sample_dimensions(n_samples, sigma_factor)
        
        results = []
        
        for i, sample in enumerate(parameter_samples):
            if i % 100 == 0:
                print(f"  Processing sample {i+1}/{n_samples}")
            
            # Start with the parameter sample
            result_row = {'sample_id': i}
            result_row.update(sample)
            
            # Calculate all measurements for this sample
            for measurement in self.measurements:
                try:
                    measurement_result = self.calculate_measurement(measurement, sample)
                    
                    # Add measurement results with prefixed names
                    for key, value in measurement_result.items():
                        result_row[f"{measurement.name}_{key}"] = value
                        
                except Exception as e:
                    print(f"Warning: Failed to calculate {measurement.name} for sample {i}: {e}")
                    # Add NaN values for failed measurements
                    if measurement.measurement_type == MeasurementType.DISTANCE:
                        result_row[f"{measurement.name}_distance"] = np.nan
                    elif measurement.measurement_type == MeasurementType.POSITION_3D:
                        result_row[f"{measurement.name}_delta_x"] = np.nan
                        result_row[f"{measurement.name}_delta_y"] = np.nan
                        result_row[f"{measurement.name}_delta_z"] = np.nan
                    elif measurement.measurement_type == MeasurementType.FULL_6DOF:
                        for coord in ['delta_x', 'delta_y', 'delta_z', 'delta_rx', 'delta_ry', 'delta_rz']:
                            result_row[f"{measurement.name}_{coord}"] = np.nan
            
            results.append(result_row)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Store results
        self.results_history.append({
            'n_samples': n_samples,
            'sigma_factor': sigma_factor,
            'results': df
        })
        
        print(f"Monte Carlo analysis complete. Generated {len(df)} samples.")
        return df
    
    def export_results(self, df: pd.DataFrame, filename: str):
        """Export Monte Carlo results to CSV file"""
        filepath = Path(filename)
        df.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")
        
        # Also create a summary statistics file
        summary_file = filepath.with_suffix('.summary.csv')
        
        # Calculate summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary_stats = df[numeric_cols].describe()
        summary_stats.to_csv(summary_file)
        print(f"Summary statistics exported to {summary_file}")
    
    def analyze_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze Monte Carlo results and provide insights.
        
        Returns:
            Dictionary with analysis results and insights
        """
        analysis = {
            'n_samples': len(df),
            'toleranced_dimensions': len(self.toleranced_dimensions),
            'measurements': len(self.measurements),
            'dimension_statistics': {},
            'measurement_statistics': {},
            'correlations': {},
            'insights': []
        }
        
        # Analyze dimensional variations
        for dim_info in self.toleranced_dimensions:
            param_names = dim_info.get('parameter_names', [])
            for param_name in param_names:
                if param_name in df.columns:
                    param_data = df[param_name].dropna()
                    analysis['dimension_statistics'][param_name] = {
                        'mean': param_data.mean(),
                        'std': param_data.std(),
                        'min': param_data.min(),
                        'max': param_data.max(),
                        'range': param_data.max() - param_data.min()
                    }
        
        # Analyze measurement variations
        measurement_cols = [col for col in df.columns if any(m.name in col for m in self.measurements)]
        for col in measurement_cols:
            if col in df.columns:
                data = df[col].dropna()
                if len(data) > 0:
                    analysis['measurement_statistics'][col] = {
                        'mean': data.mean(),
                        'std': data.std(),
                        'min': data.min(),
                        'max': data.max(),
                        'range': data.max() - data.min()
                    }
        
        # Find correlations between dimensions and measurements
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        # Extract significant correlations
        for measurement in self.measurements:
            measurement_cols = [col for col in df.columns if measurement.name in col]
            for meas_col in measurement_cols:
                if meas_col in correlation_matrix.columns:
                    correlations = correlation_matrix[meas_col].abs().sort_values(ascending=False)
                    # Get top correlated dimensions (excluding self-correlation)
                    top_correlations = correlations[correlations.index != meas_col][:5]
                    analysis['correlations'][meas_col] = top_correlations.to_dict()
        
        # Generate insights
        if analysis['measurement_statistics']:
            most_variable_measurement = max(
                analysis['measurement_statistics'].items(),
                key=lambda x: x[1]['std']
            )
            analysis['insights'].append(
                f"Most variable measurement: {most_variable_measurement[0]} "
                f"(σ = {most_variable_measurement[1]['std']:.4f})"
            )
        
        if analysis['dimension_statistics']:
            most_variable_dimension = max(
                analysis['dimension_statistics'].items(),
                key=lambda x: x[1]['std']
            )
            analysis['insights'].append(
                f"Most variable dimension: {most_variable_dimension[0]} "
                f"(σ = {most_variable_dimension[1]['std']:.4f})"
            )
        
        return analysis
    
    def print_analysis_summary(self, analysis: Dict[str, Any]):
        """Print a formatted summary of the Monte Carlo analysis"""
        print(f"\n=== Monte Carlo Analysis Summary ===")
        print(f"Samples: {analysis['n_samples']}")
        print(f"Toleranced dimensions: {analysis['toleranced_dimensions']}")
        print(f"Measurements: {analysis['measurements']}")
        
        print(f"\nMeasurement Statistics:")
        for meas_name, stats in analysis['measurement_statistics'].items():
            print(f"  {meas_name}:")
            print(f"    Mean: {stats['mean']:.4f}")
            print(f"    Std:  {stats['std']:.4f}")
            print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        print(f"\nKey Insights:")
        for insight in analysis['insights']:
            print(f"  • {insight}")
        
        print(f"\nTop Correlations:")
        for meas_name, correlations in analysis['correlations'].items():
            print(f"  {meas_name}:")
            for dim_name, corr_value in list(correlations.items())[:3]:
                print(f"    {dim_name}: {corr_value:.3f}")