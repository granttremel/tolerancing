#!/usr/bin/env python3
"""
Simple Monte Carlo test to verify the tolerance propagation is working.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tolerancing.component import Component
from tolerancing.assembly import Assembly, MateType
from tolerancing.datum import Datum, Tolerance, ToleranceType
from tolerancing.geometry import GeometryType
from tolerancing.monte_carlo import MonteCarloAnalyzer, Measurement, MeasurementType

def main():
    print("=== Simple Monte Carlo Test ===\n")
    
    # Create a very simple assembly
    base = Component("BASE")
    stage = Component("STAGE")
    
    # Base with primary datum
    base_plane_id, _ = base.add_primary_datums()
    
    # Stage with toleranced offset
    stage_plane_id, _ = stage.add_primary_datums()
    
    # Create a simple toleranced dimension
    ref_height_id = base.create_derived_datum(
        reference_id=base_plane_id,
        dimension_type='offset_distance',
        values=[10.0],
        target_geometry=GeometryType.PLANE,
        constraints={'direction': [0, 0, 1]},
        datum_id="HEIGHT",
        name="HEIGHT_10MM"
    )
    
    # Add a reasonable tolerance
    height_datum = base.get_datum(ref_height_id)
    if height_datum.dimension:
        height_datum.dimension.set_tolerance(Tolerance(ToleranceType.DIMENSION, 0.1, 0.1))  # ±0.1mm
    
    # Create assembly
    assembly = Assembly("SIMPLE_TEST")
    assembly.add_component(base, ground=True)
    assembly.add_component(stage)
    
    # Mate the components
    assembly.mate_components("BASE", "HEIGHT", "STAGE", "A", MateType.COINCIDENT)
    
    print(f"Assembly: {assembly}")
    
    # Set up Monte Carlo
    mc = MonteCarloAnalyzer(assembly)
    
    print(f"Toleranced dimensions found: {len(mc.toleranced_dimensions)}")
    for dim in mc.toleranced_dimensions:
        print(f"  {dim}")
    
    # Add a simple measurement
    measurement = Measurement(
        name="HEIGHT_TEST",
        measurement_type=MeasurementType.DISTANCE,
        source_component="BASE",
        source_datum="A",
        target_component="BASE", 
        target_datum="HEIGHT",
        description="Distance from base to height datum"
    )
    
    mc.add_measurement(measurement)
    
    # Run small Monte Carlo
    results = mc.run_monte_carlo(n_samples=50, sigma_factor=3.0)
    
    print(f"\nResults shape: {results.shape}")
    print(f"Sample of results:")
    print(results[['HEIGHT_TEST_distance']].describe())
    
    # Check if we're getting variation
    distance_data = results['HEIGHT_TEST_distance']
    print(f"\nDistance statistics:")
    print(f"  Mean: {distance_data.mean():.6f}")
    print(f"  Std:  {distance_data.std():.6f}")
    print(f"  Min:  {distance_data.min():.6f}")
    print(f"  Max:  {distance_data.max():.6f}")
    
    if distance_data.std() > 0.001:
        print("✓ Tolerance propagation is working!")
    else:
        print("✗ No variation detected - tolerance propagation issue")
        
        # Debug: Check the sampled parameters
        print(f"\nDebugging sampled parameters:")
        param_cols = [col for col in results.columns if col.startswith('d_')]
        if param_cols:
            for col in param_cols[:3]:  # Show first 3 parameter columns
                data = results[col]
                print(f"  {col}: mean={data.mean():.6f}, std={data.std():.6f}")

if __name__ == "__main__":
    main()