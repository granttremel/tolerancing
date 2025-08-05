#!/usr/bin/env python3
"""
Complete Monte Carlo tolerance analysis example.

This demonstrates:
1. Setting up an assembly with multiple toleranced dimensions
2. Defining measurements between datums (full 6-DOF relationships)
3. Running Monte Carlo simulation with normal distributions (±3σ)
4. Exporting results to CSV files
5. Statistical analysis of tolerance effects
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from tolerancing.component import Component
from tolerancing.assembly import Assembly, MateType
from tolerancing.datum import Datum, Tolerance, ToleranceType
from tolerancing.geometry import GeometryType
from tolerancing.monte_carlo import MonteCarloAnalyzer, Measurement, MeasurementType

def create_optical_assembly():
    """Create a complex optical assembly for Monte Carlo analysis"""
    
    # ===== BASE COMPONENT =====
    base = Component("BASE")
    
    # Primary datums
    base_plane_id, base_axis_id = base.add_primary_datums(
        plane_origin=[0, 0, 0],
        plane_normal=[0, 0, 1]
    )
    
    # Reference height with tight tolerance
    ref_height_id = base.create_derived_datum(
        reference_id=base_plane_id,
        dimension_type='offset_distance',
        values=[50.0],
        target_geometry=GeometryType.PLANE,
        constraints={'direction': [0, 0, 1]},
        datum_id="REF_HEIGHT",
        name="BASE_REF_HEIGHT"
    )
    
    # Add tight tolerance (±0.005mm = ±3σ, so σ = 1.67μm)
    ref_height_datum = base.get_datum(ref_height_id)
    if ref_height_datum.dimension:
        ref_height_datum.dimension.set_tolerance(Tolerance(ToleranceType.DIMENSION, 0.005, 0.005))
    
    # Mounting holes with position tolerances
    mount_positions = [(30, 0), (0, 30), (-30, 0), (0, -30)]
    for i, (x, y) in enumerate(mount_positions):
        mount_id = base.create_derived_datum(
            reference_id=ref_height_id,
            dimension_type='offset_xy',
            values=[x, y],
            target_geometry=GeometryType.POINT,
            datum_id=f"MOUNT_{i+1}",
            name=f"BASE_MOUNT_{i+1}"
        )
        
        mount_datum = base.get_datum(mount_id)
        mount_datum.set_tolerance(Tolerance(ToleranceType.POSITION, 0.02))  # ±20μm position tolerance
    
    # ===== STAGE COMPONENT =====
    stage = Component("STAGE")
    
    stage_plane_id, stage_axis_id = stage.add_primary_datums(
        plane_origin=[0, 0, 0],
        plane_normal=[0, 0, -1]  # Bottom face
    )
    
    # Stage thickness with medium tolerance
    stage_top_id = stage.create_derived_datum(
        reference_id=stage_plane_id,
        dimension_type='offset_distance',
        values=[15.0],
        target_geometry=GeometryType.PLANE,
        constraints={'direction': [0, 0, 1]},
        datum_id="STAGE_TOP",
        name="STAGE_TOP_SURFACE"
    )
    
    stage_top_datum = stage.get_datum(stage_top_id)
    if stage_top_datum.dimension:
        stage_top_datum.dimension.set_tolerance(Tolerance(ToleranceType.DIMENSION, 0.01, 0.01))
    
    # Critical optical reference point
    opt_ref_id = stage.create_derived_datum(
        reference_id=stage_top_id,
        dimension_type='offset_xy',
        values=[0.0, 0.0],  # Centered
        target_geometry=GeometryType.POINT,
        datum_id="OPT_REF",
        name="OPTICAL_REFERENCE"
    )
    
    opt_ref_datum = stage.get_datum(opt_ref_id)
    opt_ref_datum.set_tolerance(Tolerance(ToleranceType.POSITION, 0.005))  # Very tight ±5μm
    
    # Adjustment points
    adj_positions = [(25, 25), (-25, 25), (-25, -25), (25, -25)]
    for i, (x, y) in enumerate(adj_positions):
        adj_id = stage.create_derived_datum(
            reference_id=stage_top_id,
            dimension_type='offset_xy',
            values=[x, y],
            target_geometry=GeometryType.POINT,
            datum_id=f"ADJ_{i+1}",
            name=f"ADJUSTMENT_{i+1}"
        )
        
        adj_datum = stage.get_datum(adj_id)
        adj_datum.set_tolerance(Tolerance(ToleranceType.POSITION, 0.015))
    
    # ===== LENS COMPONENT =====
    lens = Component("LENS")
    
    lens_plane_id, lens_axis_id = lens.add_primary_datums(
        plane_origin=[0, 0, 0],
        plane_normal=[0, 0, -1]  # Back surface
    )
    
    # Lens thickness - critical for optical performance
    lens_front_id = lens.create_derived_datum(
        reference_id=lens_plane_id,
        dimension_type='offset_distance',
        values=[6.0],
        target_geometry=GeometryType.PLANE,
        constraints={'direction': [0, 0, 1]},
        datum_id="LENS_FRONT",
        name="LENS_FRONT_SURFACE"
    )
    
    lens_front_datum = lens.get_datum(lens_front_id)
    if lens_front_datum.dimension:
        # Ultra-tight tolerance for optical performance
        lens_front_datum.dimension.set_tolerance(Tolerance(ToleranceType.DIMENSION, 0.001, 0.001))
    
    # Optical center
    opt_center_id = lens.create_derived_datum(
        reference_id=lens_plane_id,
        dimension_type='offset_distance',
        values=[3.0],  # Center of lens
        target_geometry=GeometryType.POINT,
        constraints={'direction': [0, 0, 1]},
        datum_id="OPT_CENTER",
        name="OPTICAL_CENTER"
    )
    
    return base, stage, lens

def setup_assembly_and_measurements():
    """Set up the assembly and define measurements"""
    
    # Create components
    base, stage, lens = create_optical_assembly()
    
    # Create assembly
    assembly = Assembly("PRECISION_OPTICAL_SYSTEM")
    assembly.add_component(base, ground=True)
    assembly.add_component(stage)
    assembly.add_component(lens)
    
    # Mate components
    assembly.mate_components("BASE", "REF_HEIGHT", "STAGE", "A", MateType.COINCIDENT)
    assembly.mate_components("STAGE", "STAGE_TOP", "LENS", "A", MateType.COINCIDENT)
    assembly.mate_components("BASE", "B", "LENS", "B", MateType.CONCENTRIC)
    
    # Set up Monte Carlo analyzer
    mc_analyzer = MonteCarloAnalyzer(assembly)
    
    # Define critical measurements
    measurements = [
        # Distance from base reference to optical center
        Measurement(
            name="BASE_TO_OPTICAL_CENTER",
            measurement_type=MeasurementType.DISTANCE,
            source_component="BASE",
            source_datum="REF_HEIGHT",
            target_component="LENS",
            target_datum="OPT_CENTER",
            description="Distance from base reference to lens optical center"
        ),
        
        # Full 6-DOF relationship between optical references
        Measurement(
            name="OPTICAL_ALIGNMENT",
            measurement_type=MeasurementType.FULL_6DOF,
            source_component="STAGE",
            source_datum="OPT_REF",
            target_component="LENS",
            target_datum="OPT_CENTER",
            description="6-DOF alignment between stage and lens optical references"
        ),
        
        # Position difference between stage and lens references
        Measurement(
            name="STAGE_LENS_POSITION",
            measurement_type=MeasurementType.POSITION_3D,
            source_component="STAGE",
            source_datum="OPT_REF",
            target_component="LENS",
            target_datum="OPT_CENTER",
            description="3D position relationship between stage and lens"
        ),
        
        # Concentricity between base axis and lens axis
        Measurement(
            name="AXIS_CONCENTRICITY",
            measurement_type=MeasurementType.CONCENTRICITY,
            source_component="BASE",
            source_datum="B",
            target_component="LENS",
            target_datum="B",
            description="Concentricity between base and lens axes"
        ),
        
        # Distance between adjustment points
        Measurement(
            name="ADJ_SPACING",
            measurement_type=MeasurementType.DISTANCE,
            source_component="STAGE",
            source_datum="ADJ_1",
            target_component="STAGE",
            target_datum="ADJ_3",
            description="Distance between opposite adjustment points"
        )
    ]
    
    # Add measurements to analyzer
    for measurement in measurements:
        mc_analyzer.add_measurement(measurement)
    
    return assembly, mc_analyzer

def main():
    print("=== Monte Carlo Tolerance Analysis ===\n")
    
    # Step 1: Set up assembly and measurements
    print("1. Setting up Assembly and Measurements:")
    assembly, mc_analyzer = setup_assembly_and_measurements()
    
    print(f"   {assembly}")
    print(f"   Toleranced dimensions found: {len(mc_analyzer.toleranced_dimensions)}")
    print(f"   Measurements defined: {len(mc_analyzer.measurements)}")
    
    # Show toleranced dimensions
    print("\n   Toleranced Dimensions:")
    for dim_info in mc_analyzer.toleranced_dimensions:
        if dim_info['type'] == 'dimension':
            tol = dim_info['tolerance']
            print(f"     {dim_info['component']}.{dim_info['datum']}: "
                  f"{dim_info['dimension_type']} ±{tol.pos:.3f}mm")
        elif dim_info['type'] == 'geometric':
            tol_val = dim_info['tolerance_value']
            print(f"     {dim_info['component']}.{dim_info['datum']}: "
                  f"position ⌖{tol_val:.3f}mm")
    
    # Show measurements
    print("\n   Measurements:")
    for measurement in mc_analyzer.measurements:
        print(f"     {measurement.name}: {measurement.description}")
    
    # Step 2: Run Monte Carlo analysis
    print(f"\n2. Running Monte Carlo Analysis:")
    n_samples = 1000  # Full analysis
    sigma_factor = 3.0  # ±3σ limits
    
    print(f"   Samples: {n_samples}")
    print(f"   Tolerance interpretation: ±3σ (99.7% of population)")
    
    # Run the analysis
    results_df = mc_analyzer.run_monte_carlo(n_samples=n_samples, sigma_factor=sigma_factor)
    
    print(f"   Results shape: {results_df.shape}")
    print(f"   Columns: {len(results_df.columns)}")
    
    # Step 3: Export results
    print(f"\n3. Exporting Results:")
    
    # Export to CSV
    csv_filename = "monte_carlo_results.csv"
    mc_analyzer.export_results(results_df, csv_filename)
    
    # Step 4: Statistical analysis
    print(f"\n4. Statistical Analysis:")
    
    analysis = mc_analyzer.analyze_results(results_df)
    mc_analyzer.print_analysis_summary(analysis)
    
    # Step 5: Key insights and visualization
    print(f"\n5. Key Performance Insights:")
    
    # Analyze critical measurements
    critical_measurements = [
        'BASE_TO_OPTICAL_CENTER_distance',
        'OPTICAL_ALIGNMENT_delta_z',
        'STAGE_LENS_POSITION_delta_x',
        'STAGE_LENS_POSITION_delta_y',
        'AXIS_CONCENTRICITY_concentricity_error'
    ]
    
    print(f"   Critical Performance Metrics:")
    for meas_col in critical_measurements:
        if meas_col in results_df.columns:
            data = results_df[meas_col].dropna()
            if len(data) > 0:
                mean_val = data.mean()
                std_val = data.std()
                print(f"     {meas_col}:")
                print(f"       Mean: {mean_val:.6f}mm")
                print(f"       Std:  {std_val:.6f}mm")
                print(f"       3σ range: ±{3*std_val:.6f}mm")
                
                # Check against typical optical requirements
                if 'distance' in meas_col.lower():
                    if 3*std_val > 0.010:  # 10μm requirement
                        print(f"       ⚠ WARNING: Exceeds ±10μm distance requirement")
                    else:
                        print(f"       ✓ Meets ±10μm distance requirement")
                elif 'concentricity' in meas_col.lower():
                    if 3*std_val > 0.005:  # 5μm concentricity requirement
                        print(f"       ⚠ WARNING: Exceeds ±5μm concentricity requirement")
                    else:
                        print(f"       ✓ Meets ±5μm concentricity requirement")
    
    # Step 6: Design recommendations
    print(f"\n6. Design Recommendations:")
    
    # Find most influential dimensions
    most_variable_dims = sorted(
        [(name, stats['std']) for name, stats in analysis['dimension_statistics'].items()],
        key=lambda x: x[1], reverse=True
    )[:3]
    
    print(f"   Most Variable Dimensions (consider tightening):")
    for dim_name, std_val in most_variable_dims:
        print(f"     {dim_name}: σ = {std_val:.6f}mm")
    
    # Show correlation insights
    print(f"\n   Strongest Correlations (dimension → measurement):")
    for meas_name, correlations in analysis['correlations'].items():
        if correlations:
            strongest_corr = max(correlations.items(), key=lambda x: abs(x[1]))
            if abs(strongest_corr[1]) > 0.5:  # Significant correlation
                print(f"     {strongest_corr[0]} → {meas_name}: r = {strongest_corr[1]:.3f}")
    
    # Step 7: Summary
    print(f"\n=== Monte Carlo Analysis Complete ===")
    print(f"Generated {n_samples} samples with complete tolerance propagation")
    print(f"Results exported to: {csv_filename}")
    print(f"Summary statistics in: monte_carlo_results.summary.csv")
    print(f"\nThis analysis provides:")
    print(f"• Statistical prediction of assembly performance")
    print(f"• Identification of critical tolerance contributors")
    print(f"• Design guidance for tolerance allocation")
    print(f"• Verification against optical performance requirements")
    
    # Optional: Create some simple plots if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        print(f"\n8. Creating Visualization:")
        
        # Plot distribution of critical measurement
        if 'BASE_TO_OPTICAL_CENTER_distance' in results_df.columns:
            plt.figure(figsize=(10, 6))
            try:
                plt.subplot(2, 2, 1)
                distance_data = results_df['BASE_TO_OPTICAL_CENTER_distance'].dropna()
                plt.hist(distance_data, bins=50, alpha=0.7, edgecolor='black')
                plt.xlabel('Distance (mm)')
                plt.ylabel('Frequency')
                plt.title('Base to Optical Center Distance Distribution')
                plt.grid(True, alpha=0.3)
            except Exception as e:
                print(str(e))
            
            try:
                plt.subplot(2, 2, 2)
                if 'OPTICAL_ALIGNMENT_delta_z' in results_df.columns:
                    z_data = results_df['OPTICAL_ALIGNMENT_delta_z'].dropna()
                    plt.hist(z_data, bins=50, alpha=0.7, color='orange', edgecolor='black')
                    plt.xlabel('Z Alignment Error (mm)')
                    plt.ylabel('Frequency')
                    plt.title('Optical Z-Alignment Distribution')
                    plt.grid(True, alpha=0.3)
            except Exception as e:
                print(str(e))
            
            try:
                plt.subplot(2, 2, 3)
                if 'AXIS_CONCENTRICITY_concentricity_error' in results_df.columns:
                    conc_data = results_df['AXIS_CONCENTRICITY_concentricity_error'].dropna()
                    plt.hist(conc_data, bins=50, alpha=0.7, color='green', edgecolor='black')
                    plt.xlabel('Concentricity Error (mm)')
                    plt.ylabel('Frequency')
                    plt.title('Axis Concentricity Distribution')
                    plt.grid(True, alpha=0.3)
            except Exception as e:
                print(str(e))
                
            try:
                plt.subplot(2, 2, 4)
                # Scatter plot of two key measurements
                if ('STAGE_LENS_POSITION_delta_x' in results_df.columns and 
                    'STAGE_LENS_POSITION_delta_y' in results_df.columns):
                    x_data = results_df['STAGE_LENS_POSITION_delta_x'].dropna()
                    y_data = results_df['STAGE_LENS_POSITION_delta_y'].dropna()
                    plt.scatter(x_data, y_data, alpha=0.5, s=1)
                    plt.xlabel('X Position Error (mm)')
                    plt.ylabel('Y Position Error (mm)')
                    plt.title('XY Position Error Correlation')
                    plt.grid(True, alpha=0.3)
                    plt.axis('equal')
            except Exception as e:
                print(str(e))
            plt.tight_layout()
            plt.savefig('monte_carlo_analysis.png', dpi=150, bbox_inches='tight')
            print(f"   Plots saved to: monte_carlo_analysis.png")
            
        # Show the plots
        # plt.show()  # Uncomment to display plots
        
    except ImportError:
        print(f"   Matplotlib not available - skipping visualization")

if __name__ == "__main__":
    main()