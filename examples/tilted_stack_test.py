#!/usr/bin/env python3
"""
Test script for planar component stack with deliberate tilt.

This creates an assembly with multiple planar components stacked in series,
with one component deliberately tilted by ~5 degrees to test:
1. Mate constraint handling for tilted planes
2. Coordinate transformation propagation through the stack
3. Geometric relationship calculations
4. Assembly positioning and constraint solving

The stack consists of:
- Base plate (ground component)
- Spacer 1 (normal orientation)  
- Tilted plate (5-degree tilt)
- Spacer 2 (follows tilt)
- Top plate (final component)
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

def create_planar_component(name: str, thickness: float, is_tilted: bool = False, tilt_angle_deg: float = 0.0):
    """
    Create a planar component with specified thickness and optional tilt.
    
    Args:
        name: Component name
        thickness: Component thickness in mm
        is_tilted: Whether to apply a deliberate tilt
        tilt_angle_deg: Tilt angle in degrees (if is_tilted=True)
    """
    component = Component(name)
    
    if is_tilted:
        # Create tilted primary datums
        tilt_rad = np.radians(tilt_angle_deg)
        
        # Tilted plane normal (rotate around Y-axis)
        tilted_normal = np.array([
            np.sin(tilt_rad),  # X component
            0.0,               # Y component  
            np.cos(tilt_rad)   # Z component
        ])
        
        print(f"Creating {name} with {tilt_angle_deg}° tilt")
        print(f"  Normal vector: [{tilted_normal[0]:.4f}, {tilted_normal[1]:.4f}, {tilted_normal[2]:.4f}]")
        
        # Primary datums with tilt
        bottom_plane_id, axis_id = component.add_primary_datums(
            plane_origin=[0, 0, 0],
            plane_normal=tilted_normal.tolist()
        )
    else:
        # Standard horizontal plane
        bottom_plane_id, axis_id = component.add_primary_datums(
            plane_origin=[0, 0, 0],
            plane_normal=[0, 0, 1]
        )
    
    # Create top surface at specified thickness
    if is_tilted:
        # For tilted component, thickness is measured along the tilted normal
        tilt_rad = np.radians(tilt_angle_deg)
        tilted_normal = np.array([np.sin(tilt_rad), 0.0, np.cos(tilt_rad)])
        
        top_plane_id = component.create_derived_datum(
            reference_id=bottom_plane_id,
            dimension_type='offset_distance',
            values=[thickness],
            target_geometry=GeometryType.PLANE,
            constraints={'direction': tilted_normal.tolist()},
            datum_id="TOP",
            name=f"{name}_TOP_SURFACE"
        )
    else:
        # Standard vertical offset
        top_plane_id = component.create_derived_datum(
            reference_id=bottom_plane_id,
            dimension_type='offset_distance',
            values=[thickness],
            target_geometry=GeometryType.PLANE,
            constraints={'direction': [0, 0, 1]},
            datum_id="TOP",
            name=f"{name}_TOP_SURFACE"
        )
    
    # Add dimensional tolerance to thickness
    top_datum = component.get_datum(top_plane_id)
    if top_datum.dimension:
        top_datum.dimension.set_tolerance(Tolerance(ToleranceType.DIMENSION, 0.05, 0.05))  # ±50μm
    
    # Add reference points for measurements
    center_points = []
    for surface, surface_id in [("BOTTOM", bottom_plane_id), ("TOP", top_plane_id)]:
        center_id = component.create_derived_datum(
            reference_id=surface_id,
            dimension_type='offset_xy',
            values=[0.0, 0.0],
            target_geometry=GeometryType.POINT,
            datum_id=f"{surface}_CENTER",
            name=f"{name}_{surface}_CENTER"
        )
        center_points.append(center_id)
    
    # Add corner points for tilt analysis
    corner_positions = [
        ("NE", [10.0, 10.0]),   # Northeast corner
        ("NW", [-10.0, 10.0]),  # Northwest corner  
        ("SE", [10.0, -10.0]),  # Southeast corner
        ("SW", [-10.0, -10.0])  # Southwest corner
    ]
    
    for corner_name, (x, y) in corner_positions:
        for surface, surface_id in [("BOTTOM", bottom_plane_id), ("TOP", top_plane_id)]:
            corner_id = component.create_derived_datum(
                reference_id=surface_id,
                dimension_type='offset_xy',
                values=[x, y],
                target_geometry=GeometryType.POINT,
                datum_id=f"{surface}_{corner_name}",
                name=f"{name}_{surface}_{corner_name}"
            )
    
    return component

def create_tilted_stack_assembly():
    """Create an assembly with a stack of planar components including one tilted component"""
    
    print("=== Creating Tilted Stack Assembly ===\n")
    
    # Component specifications
    component_specs = [
        ("BASE_PLATE", 10.0, False, 0.0),      # Base plate: 10mm thick
        ("SPACER_1", 5.0, False, 0.0),         # Spacer 1: 5mm thick  
        ("TILTED_PLATE", 8.0, True, 5.0),      # Tilted plate: 8mm thick, 5° tilt
        ("SPACER_2", 5.0, False, 0.0),         # Spacer 2: 5mm thick
        ("TOP_PLATE", 6.0, False, 0.0)         # Top plate: 6mm thick
    ]
    
    # Create all components
    components = {}
    for name, thickness, is_tilted, tilt_angle in component_specs:
        component = create_planar_component(name, thickness, is_tilted, tilt_angle)
        components[name] = component
        
        print(f"Created {name}: {thickness}mm thick" + 
              (f", {tilt_angle}° tilt" if is_tilted else ""))
    
    print(f"\nTotal components created: {len(components)}")
    
    # Create assembly
    assembly = Assembly("TILTED_STACK")
    
    # Add components to assembly
    assembly.add_component(components["BASE_PLATE"], ground=True)  # Ground component
    for name in ["SPACER_1", "TILTED_PLATE", "SPACER_2", "TOP_PLATE"]:
        assembly.add_component(components[name])
    
    print(f"\nAssembly created with {len(assembly.components)} components")
    
    # Create mate relationships (stack from bottom to top)
    mate_sequence = [
        ("BASE_PLATE", "TOP", "SPACER_1", "A"),           # Base top → Spacer 1 bottom
        ("SPACER_1", "TOP", "TILTED_PLATE", "A"),         # Spacer 1 top → Tilted bottom  
        ("TILTED_PLATE", "TOP", "SPACER_2", "A"),         # Tilted top → Spacer 2 bottom
        ("SPACER_2", "TOP", "TOP_PLATE", "A")             # Spacer 2 top → Top plate bottom
    ]
    
    print(f"\nCreating mate relationships:")
    for i, (comp1, datum1, comp2, datum2) in enumerate(mate_sequence):
        try:
            assembly.mate_components(comp1, datum1, comp2, datum2, MateType.COINCIDENT)
            print(f"  {i+1}. {comp1}.{datum1} ↔ {comp2}.{datum2} (COINCIDENT)")
        except Exception as e:
            print(f"  {i+1}. FAILED: {comp1}.{datum1} ↔ {comp2}.{datum2} - {e}")
    
    print(f"\nTotal mates: {len(assembly.mates)}")
    
    return assembly, components

def setup_stack_measurements(assembly):
    """Set up measurements to analyze the tilted stack"""
    
    mc_analyzer = MonteCarloAnalyzer(assembly)
    
    # Define measurements through the stack
    measurements = [
        # Overall stack height
        Measurement(
            name="TOTAL_STACK_HEIGHT",
            measurement_type=MeasurementType.DISTANCE,
            source_component="BASE_PLATE",
            source_datum="BOTTOM_CENTER", 
            target_component="TOP_PLATE",
            target_datum="TOP_CENTER",
            description="Total height of component stack"
        ),
        
        # Height to tilted plate center
        Measurement(
            name="BASE_TO_TILTED_CENTER",
            measurement_type=MeasurementType.DISTANCE,
            source_component="BASE_PLATE",
            source_datum="BOTTOM_CENTER",
            target_component="TILTED_PLATE", 
            target_datum="BOTTOM_CENTER",
            description="Height from base to tilted plate center"
        ),
        
        # 6-DOF relationship showing tilt effects
        Measurement(
            name="BASE_TO_TILTED_ALIGNMENT",
            measurement_type=MeasurementType.FULL_6DOF,
            source_component="BASE_PLATE",
            source_datum="BOTTOM_CENTER",
            target_component="TILTED_PLATE",
            target_datum="BOTTOM_CENTER", 
            description="Complete alignment from base to tilted plate"
        ),
        
        # Corner height measurements to quantify tilt
        Measurement(
            name="TILTED_NE_CORNER_HEIGHT",
            measurement_type=MeasurementType.DISTANCE,
            source_component="BASE_PLATE",
            source_datum="BOTTOM_CENTER",
            target_component="TILTED_PLATE",
            target_datum="TOP_NE",
            description="Height to tilted plate NE corner"
        ),
        
        Measurement(
            name="TILTED_SW_CORNER_HEIGHT", 
            measurement_type=MeasurementType.DISTANCE,
            source_component="BASE_PLATE",
            source_datum="BOTTOM_CENTER",
            target_component="TILTED_PLATE",
            target_datum="TOP_SW",
            description="Height to tilted plate SW corner"
        ),
        
        # Final stack alignment
        Measurement(
            name="BASE_TO_TOP_ALIGNMENT",
            measurement_type=MeasurementType.FULL_6DOF,
            source_component="BASE_PLATE", 
            source_datum="BOTTOM_CENTER",
            target_component="TOP_PLATE",
            target_datum="TOP_CENTER",
            description="Complete alignment from base to top plate"
        ),
        
        # Cross-stack diagonal measurement
        Measurement(
            name="DIAGONAL_STACK_MEASUREMENT",
            measurement_type=MeasurementType.POSITION_3D,
            source_component="BASE_PLATE",
            source_datum="BOTTOM_NE",
            target_component="TOP_PLATE", 
            target_datum="TOP_SW",
            description="Diagonal measurement across entire stack"
        )
    ]
    
    # Add measurements to analyzer
    for measurement in measurements:
        mc_analyzer.add_measurement(measurement)
    
    return mc_analyzer

def analyze_tilt_effects(results_df):
    """Analyze the effects of the tilted component on the stack"""
    
    print(f"\n=== Tilt Effects Analysis ===")
    
    # Analyze corner height differences (should show tilt)
    if ('TILTED_NE_CORNER_HEIGHT_distance' in results_df.columns and 
        'TILTED_SW_CORNER_HEIGHT_distance' in results_df.columns):
        
        ne_heights = results_df['TILTED_NE_CORNER_HEIGHT_distance'].dropna()
        sw_heights = results_df['TILTED_SW_CORNER_HEIGHT_distance'].dropna()
        
        if len(ne_heights) > 0 and len(sw_heights) > 0:
            ne_mean = ne_heights.mean()
            sw_mean = sw_heights.mean()
            height_diff = ne_mean - sw_mean
            
            print(f"\nTilted Plate Corner Analysis:")
            print(f"  NE corner height: {ne_mean:.4f} mm")
            print(f"  SW corner height: {sw_mean:.4f} mm") 
            print(f"  Height difference: {height_diff:.4f} mm")
            
            # Calculate implied tilt angle
            corner_separation = np.sqrt((20)**2 + (20)**2)  # Diagonal distance = ~28.28mm
            tilt_angle_rad = np.arctan(height_diff / corner_separation)
            tilt_angle_deg = np.degrees(tilt_angle_rad)
            
            print(f"  Diagonal separation: {corner_separation:.2f} mm")
            print(f"  Measured tilt angle: {tilt_angle_deg:.2f}°")
            print(f"  Expected tilt angle: 5.00°")
            
            if abs(tilt_angle_deg - 5.0) < 0.5:
                print(f"  ✓ Tilt measurement matches expected value")
            else:
                print(f"  ⚠ Tilt measurement differs from expected")
    
    # Analyze overall stack alignment
    if 'BASE_TO_TOP_ALIGNMENT_delta_z' in results_df.columns:
        z_alignment = results_df['BASE_TO_TOP_ALIGNMENT_delta_z'].dropna()
        if len(z_alignment) > 0:
            print(f"\nOverall Stack Alignment:")
            print(f"  Z-axis alignment mean: {z_alignment.mean():.4f} mm")
            print(f"  Z-axis alignment std: {z_alignment.std():.6f} mm")
            
            expected_height = 10 + 5 + 8 + 5 + 6  # Sum of all thicknesses
            print(f"  Expected total height: {expected_height} mm")
            
            if abs(z_alignment.mean() - expected_height) < 0.1:
                print(f"  ✓ Total height matches expected value")
            else:
                print(f"  ⚠ Total height differs from expected")
    
    # Analyze angular deviations
    angular_cols = ['BASE_TO_TOP_ALIGNMENT_delta_rx', 'BASE_TO_TOP_ALIGNMENT_delta_ry']
    
    print(f"\nAngular Deviation Analysis:")
    for col in angular_cols:
        if col in results_df.columns:
            data = results_df[col].dropna()
            if len(data) > 0:
                axis = col.split('_')[-1]  # rx or ry
                print(f"  {axis.upper()} rotation: {data.mean():.6f} rad ({np.degrees(data.mean()):.4f}°)")
                print(f"  {axis.upper()} std: {data.std():.6f} rad ({np.degrees(data.std()):.4f}°)")

def main():
    print("=== Tilted Stack Test Script ===\n")
    
    # Step 1: Create the tilted stack assembly
    print("1. Creating Tilted Stack Assembly:")
    assembly, components = create_tilted_stack_assembly()
    
    for cn,c in assembly.components.items():
        print(cn)
        for dn,d in c.datums.items():
            print(dn,d.geo.aorigin, d.geo.aframe)
    
    
    # Step 2: Set up measurements
    print(f"\n2. Setting up Measurements:")
    mc_analyzer = setup_stack_measurements(assembly)
    
    print(f"   Measurements defined: {len(mc_analyzer.measurements)}")
    print(f"   Toleranced dimensions: {len(mc_analyzer.toleranced_dimensions)}")
    
    # Show the measurements
    print(f"\n   Measurement List:")
    for i, measurement in enumerate(mc_analyzer.measurements):
        print(f"     {i+1}. {measurement.name}: {measurement.description}")
    
    # Step 3: Show assembly structure
    print(f"\n3. Assembly Structure:")
    print(f"   Components: {len(assembly.components)}")
    print(f"   Mates: {len(assembly.mates)}")
    
    for mate in assembly.mates:
        print(f"     {mate.component1.name}.{mate.datum1_id} ↔ " +
              f"{mate.component2.name}.{mate.datum2_id} ({mate.mate_type.value})")
    
    # Step 4: Test coordinate transformations
    print(f"\n4. Testing Coordinate Transformations:")
    
    try:
        # Test transformation chain for a measurement through the tilted component
        test_measurement = mc_analyzer.measurements[2]  # BASE_TO_TILTED_ALIGNMENT
        
        print(f"   Testing: {test_measurement.name}")
        print(f"   Source: {test_measurement.source_component}.{test_measurement.source_datum}")
        print(f"   Target: {test_measurement.target_component}.{test_measurement.target_datum}")
        
        # Get transformation chains
        source_chain = assembly.get_datum_global_transform(
            test_measurement.source_component, test_measurement.source_datum
        )
        target_chain = assembly.get_datum_global_transform(
            test_measurement.target_component, test_measurement.target_datum
        )
        
        print(f"   Source chain steps: {len(source_chain)}")
        print(f"   Target chain steps: {len(target_chain)}")
        
        # Test a single measurement calculation
        dummy_sample = {}  # Empty sample for nominal values
        measurement_result = mc_analyzer.calculate_measurement(test_measurement, dummy_sample)
        
        print(f"   Nominal measurement result:")
        for key, value in measurement_result.items():
            if 'delta' in key:
                print(f"     {key}: {value:.4f}")
            else:
                print(f"     {key}: {value:.4f}")
        
    except Exception as e:
        print(f"   Error in transformation test: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 5: Run a small Monte Carlo sample
    print(f"\n5. Running Small Monte Carlo Test:")
    
    try:
        n_samples = 50  # Small sample for testing
        results_df = mc_analyzer.run_monte_carlo(n_samples=n_samples, sigma_factor=3.0)
        
        print(f"   Results shape: {results_df.shape}")
        print(f"   Columns: {len(results_df.columns)}")
        
        # Analyze the results
        analyze_tilt_effects(results_df)
        
        # Export results
        csv_filename = "tilted_stack_test_results.csv"
        mc_analyzer.export_results(results_df, csv_filename)
        print(f"\n   Results exported to: {csv_filename}")
        
    except Exception as e:
        print(f"   Error in Monte Carlo analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 6: Summary
    print(f"\n=== Tilted Stack Test Complete ===")
    print(f"This test demonstrates:")
    print(f"• Planar component stack assembly")
    print(f"• Deliberate 5° tilt in middle component")
    print(f"• Mate constraint handling for tilted planes")
    print(f"• Coordinate transformation through complex geometry")
    print(f"• Tilt propagation and measurement effects")
    print(f"\nKey insights:")
    print(f"• Tilted components affect entire stack geometry")
    print(f"• Corner measurements reveal tilt angles") 
    print(f"• Mate constraints properly handle angular misalignment")
    print(f"• Transformation chains propagate tilt effects")

if __name__ == "__main__":
    main()