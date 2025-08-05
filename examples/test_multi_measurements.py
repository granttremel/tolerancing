#!/usr/bin/env python3
"""
Test multiple measurements to ensure all show proper variation.
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
    print("=== Testing Multiple Measurements ===\n")
    
    # Create a simple assembly with multiple toleranced dimensions
    base = Component("BASE")
    stage = Component("STAGE")
    
    # Base datums
    base_plane_id, base_axis_id = base.add_primary_datums()
    
    # Create height dimension
    height_id = base.create_derived_datum(
        reference_id=base_plane_id,
        dimension_type='offset_distance',
        values=[10.0],
        target_geometry=GeometryType.PLANE,
        constraints={'direction': [0, 0, 1]},
        datum_id="HEIGHT",
        name="HEIGHT_10MM"
    )
    height_datum = base.get_datum(height_id)
    if height_datum.dimension:
        height_datum.dimension.set_tolerance(Tolerance(ToleranceType.DIMENSION, 0.1, 0.1))
    
    # Create offset feature
    offset_id = base.create_derived_datum(
        reference_id=height_id,
        dimension_type='offset_xy',
        values=[20.0, 0.0],
        target_geometry=GeometryType.POINT,
        datum_id="OFFSET",
        name="OFFSET_POINT"
    )
    offset_datum = base.get_datum(offset_id)
    offset_datum.set_tolerance(Tolerance(ToleranceType.POSITION, 0.05))
    
    # Stage datums
    stage_plane_id, stage_axis_id = stage.add_primary_datums()
    
    # Create stage height
    stage_height_id = stage.create_derived_datum(
        reference_id=stage_plane_id,
        dimension_type='offset_distance',
        values=[5.0],
        target_geometry=GeometryType.PLANE,
        constraints={'direction': [0, 0, 1]},
        datum_id="STAGE_TOP",
        name="STAGE_TOP_5MM"
    )
    stage_height_datum = stage.get_datum(stage_height_id)
    if stage_height_datum.dimension:
        stage_height_datum.dimension.set_tolerance(Tolerance(ToleranceType.DIMENSION, 0.05, 0.05))
    
    # Assembly
    assembly = Assembly("TEST_MULTI")
    assembly.add_component(base, ground=True)
    assembly.add_component(stage)
    
    # Mate stage to base height
    assembly.mate_components("BASE", "HEIGHT", "STAGE", "A", MateType.COINCIDENT)
    
    # Set up Monte Carlo
    mc = MonteCarloAnalyzer(assembly)
    
    print(f"Toleranced dimensions: {len(mc.toleranced_dimensions)}")
    for dim in mc.toleranced_dimensions:
        name = dim.get('dimension_name', dim.get('datum_name', 'unnamed'))
        print(f"  {name}: {dim['dimension_type'] if 'dimension_type' in dim else dim['type']}")
    
    # Add multiple measurements
    measurements = [
        Measurement(
            name="BASE_HEIGHT",
            measurement_type=MeasurementType.DISTANCE,
            source_component="BASE",
            source_datum="A",
            target_component="BASE",
            target_datum="HEIGHT",
            description="Base to height distance"
        ),
        
        Measurement(
            name="OFFSET_POSITION", 
            measurement_type=MeasurementType.POSITION_3D,
            source_component="BASE",
            source_datum="A",
            target_component="BASE",
            target_datum="OFFSET",
            description="Offset point position"
        ),
        
        Measurement(
            name="TOTAL_HEIGHT",
            measurement_type=MeasurementType.DISTANCE,
            source_component="BASE",
            source_datum="A",
            target_component="STAGE",
            target_datum="STAGE_TOP",
            description="Total assembly height"
        ),
        
        Measurement(
            name="STAGE_THICKNESS",
            measurement_type=MeasurementType.DISTANCE,
            source_component="STAGE",
            source_datum="A",
            target_component="STAGE",
            target_datum="STAGE_TOP", 
            description="Stage thickness"
        )
    ]
    
    for measurement in measurements:
        mc.add_measurement(measurement)
    
    print(f"\nMeasurements: {len(mc.measurements)}")
    for m in mc.measurements:
        print(f"  {m.name}: {m.description}")
    
    # Run Monte Carlo
    results = mc.run_monte_carlo(n_samples=100, sigma_factor=3.0)
    
    print(f"\nResults shape: {results.shape}")
    
    # Check each measurement for variation
    measurement_cols = [col for col in results.columns if any(m.name in col for m in measurements)]
    
    print(f"\nMeasurement Variation Check:")
    for col in measurement_cols:
        if col in results.columns:
            data = results[col].dropna()
            if len(data) > 0:
                mean_val = data.mean()
                std_val = data.std()
                print(f"  {col}:")
                print(f"    Mean: {mean_val:.6f}")
                print(f"    Std:  {std_val:.6f}")
                print(f"    Range: [{data.min():.6f}, {data.max():.6f}]")
                
                if std_val > 0.001:
                    print(f"    ✓ Shows variation")
                else:
                    print(f"    ✗ No variation detected")
    
    # Export for inspection
    results.to_csv('multi_measurement_test.csv', index=False)
    print(f"\nResults exported to multi_measurement_test.csv")

if __name__ == "__main__":
    main()