#!/usr/bin/env python3
"""
Flatness tolerance analysis example for optical systems.

This demonstrates:
1. Creating optical surfaces with flatness tolerances
2. Modeling flatness as angular deviations in surface normals
3. Monte Carlo analysis of flatness effects on optical alignment
4. Understanding flatness impact on beam propagation and focus quality

Flatness tolerance is critical for optical elements because:
- Surface irregularities scatter light and reduce coherence
- Angular deviations affect beam pointing and focus quality
- Transmitted wavefront quality depends on surface flatness
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

def create_optical_flat_assembly():
    """Create an optical flat assembly with flatness tolerances"""
    
    # ===== OPTICAL MOUNT COMPONENT =====
    mount = Component("OPTICAL_MOUNT")
    
    # Primary datums for mount
    mount_plane_id, mount_axis_id = mount.add_primary_datums(
        plane_origin=[0, 0, 0],
        plane_normal=[0, 0, 1]
    )
    
    # Mount height
    mount_top_id = mount.create_derived_datum(
        reference_id=mount_plane_id,
        dimension_type='offset_distance',
        values=[25.0],
        target_geometry=GeometryType.PLANE,
        constraints={'direction': [0, 0, 1]},
        datum_id="MOUNT_TOP",
        name="MOUNT_TOP_SURFACE"
    )
    
    # ===== OPTICAL FLAT COMPONENT =====
    optical_flat = Component("OPTICAL_FLAT")
    
    # Primary datums - back surface of optical flat
    flat_back_id, flat_axis_id = optical_flat.add_primary_datums(
        plane_origin=[0, 0, 0],
        plane_normal=[0, 0, -1]  # Points toward mount
    )
    
    # Optical flat thickness (precise)
    flat_front_id = optical_flat.create_derived_datum(
        reference_id=flat_back_id,
        dimension_type='offset_distance',
        values=[6.35],  # 1/4" BK7 optical flat
        target_geometry=GeometryType.PLANE,
        constraints={'direction': [0, 0, 1]},
        datum_id="FLAT_FRONT",
        name="OPTICAL_SURFACE"
    )
    
    flat_front_datum = optical_flat.get_datum(flat_front_id)
    if flat_front_datum.dimension:
        # Very tight thickness tolerance for optical element
        flat_front_datum.dimension.set_tolerance(Tolerance(ToleranceType.DIMENSION, 0.002, 0.002))
    
    # Add FLATNESS tolerance to the optical surface
    # λ/10 @ 633nm = 63.3nm flatness requirement for precision optics
    flatness_tolerance_nm = 63.3  # nanometers
    flatness_tolerance_mm = flatness_tolerance_nm * 1e-6  # convert to mm
    
    flat_front_datum.set_tolerance(Tolerance(ToleranceType.FLATNESS, flatness_tolerance_mm))
    
    print(f"Optical surface flatness tolerance: {flatness_tolerance_nm:.1f} nm (λ/10 @ 633nm)")
    print(f"Equivalent angular deviation: {flatness_tolerance_mm/10.0*1000:.3f} mrad over 10mm aperture")
    
    # Reference points on optical surface for measurements
    surface_center_id = optical_flat.create_derived_datum(
        reference_id=flat_front_id,
        dimension_type='offset_xy',
        values=[0.0, 0.0],
        target_geometry=GeometryType.POINT,
        datum_id="OPT_CENTER",
        name="OPTICAL_CENTER"
    )
    
    # Edge reference points to measure flatness effects
    edge_points = [
        ("EDGE_N", [0.0, 5.0]),   # North edge
        ("EDGE_E", [5.0, 0.0]),   # East edge
        ("EDGE_S", [0.0, -5.0]),  # South edge
        ("EDGE_W", [-5.0, 0.0])   # West edge
    ]
    
    for edge_name, (x, y) in edge_points:
        edge_id = optical_flat.create_derived_datum(
            reference_id=flat_front_id,
            dimension_type='offset_xy',
            values=[x, y],
            target_geometry=GeometryType.POINT,
            datum_id=edge_name,
            name=f"EDGE_POINT_{edge_name}"
        )
    
    # ===== REFERENCE BEAM COMPONENT =====
    beam_ref = Component("BEAM_REFERENCE")
    
    # Beam reference plane (represents incoming laser beam)
    beam_plane_id, beam_axis_id = beam_ref.add_primary_datums(
        plane_origin=[0, 0, 50.0],  # 50mm from optical flat
        plane_normal=[0, 0, -1]     # Beam travels toward flat
    )
    
    # Beam center point
    beam_center_id = beam_ref.create_derived_datum(
        reference_id=beam_plane_id,
        dimension_type='offset_xy',
        values=[0.0, 0.0],
        target_geometry=GeometryType.POINT,
        datum_id="BEAM_CENTER",
        name="INCOMING_BEAM_CENTER"
    )
    
    return mount, optical_flat, beam_ref

def setup_flatness_analysis():
    """Set up assembly and measurements for flatness analysis"""
    
    # Create components
    mount, optical_flat, beam_ref = create_optical_flat_assembly()
    
    # Create assembly
    assembly = Assembly("OPTICAL_FLAT_SYSTEM")
    assembly.add_component(mount, ground=True)
    assembly.add_component(optical_flat)
    assembly.add_component(beam_ref)
    
    # Mate optical flat to mount
    assembly.mate_components("OPTICAL_MOUNT", "MOUNT_TOP", "OPTICAL_FLAT", "A", MateType.COINCIDENT)
    
    # Set up Monte Carlo analyzer
    mc_analyzer = MonteCarloAnalyzer(assembly)
    
    # Define measurements to assess flatness effects
    measurements = [
        # Distance from beam reference to optical surface center
        Measurement(
            name="BEAM_TO_SURFACE",
            measurement_type=MeasurementType.DISTANCE,
            source_component="BEAM_REFERENCE",
            source_datum="BEAM_CENTER",
            target_component="OPTICAL_FLAT",
            target_datum="OPT_CENTER",
            description="Distance from beam to optical surface center"
        ),
        
        # 6-DOF relationship between beam and optical surface
        Measurement(
            name="BEAM_SURFACE_ALIGNMENT",
            measurement_type=MeasurementType.FULL_6DOF,
            source_component="BEAM_REFERENCE",
            source_datum="BEAM_CENTER",
            target_component="OPTICAL_FLAT",
            target_datum="OPT_CENTER",
            description="Complete beam-to-surface alignment"
        ),
        
        # Surface flatness measurements (relative heights)
        Measurement(
            name="CENTER_TO_NORTH_EDGE",
            measurement_type=MeasurementType.DISTANCE,
            source_component="OPTICAL_FLAT",
            source_datum="OPT_CENTER",
            target_component="OPTICAL_FLAT",
            target_datum="EDGE_N",
            description="Center to north edge height difference"
        ),
        
        Measurement(
            name="CENTER_TO_EAST_EDGE",
            measurement_type=MeasurementType.DISTANCE,
            source_component="OPTICAL_FLAT",
            source_datum="OPT_CENTER",
            target_component="OPTICAL_FLAT",
            target_datum="EDGE_E",
            description="Center to east edge height difference"
        ),
        
        # Diagonal measurement for surface tilt
        Measurement(
            name="NORTH_TO_SOUTH_TILT",
            measurement_type=MeasurementType.POSITION_3D,
            source_component="OPTICAL_FLAT",
            source_datum="EDGE_N",
            target_component="OPTICAL_FLAT",
            target_datum="EDGE_S",
            description="North-South surface tilt"
        ),
        
        Measurement(
            name="EAST_TO_WEST_TILT",
            measurement_type=MeasurementType.POSITION_3D,
            source_component="OPTICAL_FLAT",
            source_datum="EDGE_E",
            target_component="OPTICAL_FLAT",
            target_datum="EDGE_W",
            description="East-West surface tilt"
        )
    ]
    
    # Add measurements to analyzer
    for measurement in measurements:
        mc_analyzer.add_measurement(measurement)
        assembly.print_transformation_tree(measurement.source_component, measurement.source_datum)
    
    return assembly, mc_analyzer

def analyze_flatness_effects(results_df):
    """Analyze the effects of flatness tolerance on optical performance"""
    
    print(f"\n=== Flatness Effects Analysis ===")
    
    # Analyze beam-to-surface variations
    if 'BEAM_SURFACE_ALIGNMENT_delta_z' in results_df.columns:
        z_alignment = results_df['BEAM_SURFACE_ALIGNMENT_delta_z'].dropna()
        print(f"\nBeam-Surface Z Alignment:")
        print(f"  Mean: {z_alignment.mean():.6f} mm")
        print(f"  Std:  {z_alignment.std():.6f} mm") 
        print(f"  3σ range: ±{3*z_alignment.std():.6f} mm")
        
        # Convert to optical path difference (OPD)
        # For BK7 at 633nm: n ≈ 1.515
        n_bk7 = 1.515
        opd_nm = z_alignment.std() * (n_bk7 - 1) * 1e6  # Convert to nanometers
        print(f"  Optical Path Difference: {opd_nm:.2f} nm RMS")
        
        if opd_nm > 20:  # λ/30 criterion for wavefront quality
            print(f"  ⚠ WARNING: OPD exceeds λ/30 criterion for diffraction-limited performance")
        else:
            print(f"  ✓ OPD meets λ/30 criterion for diffraction-limited performance")
    
    # Analyze surface tilt effects
    tilt_measurements = ['NORTH_TO_SOUTH_TILT_delta_z', 'EAST_TO_WEST_TILT_delta_z']
    
    print(f"\nSurface Tilt Analysis:")
    for tilt_col in tilt_measurements:
        if tilt_col in results_df.columns:
            tilt_data = results_df[tilt_col].dropna()
            if len(tilt_data) > 0:
                tilt_angle_mrad = tilt_data.std() / 10.0 * 1000  # Convert to mrad over 10mm
                tilt_angle_arcsec = tilt_angle_mrad * 206.265  # Convert to arcseconds
                
                direction = "North-South" if "NORTH" in tilt_col else "East-West"
                print(f"  {direction} Tilt:")
                print(f"    Height variation: {tilt_data.std():.6f} mm RMS")
                print(f"    Angular deviation: {tilt_angle_mrad:.3f} mrad RMS")
                print(f"    Angular deviation: {tilt_angle_arcsec:.1f} arcsec RMS")
                
                # Beam deflection analysis
                # For reflection: deflection angle = 2 × surface tilt
                beam_deflection_mrad = 2 * tilt_angle_mrad
                beam_deflection_arcsec = 2 * tilt_angle_arcsec
                
                print(f"    Reflected beam deflection: {beam_deflection_mrad:.3f} mrad RMS")
                print(f"    Reflected beam deflection: {beam_deflection_arcsec:.1f} arcsec RMS")
                
                # Spot size growth at 1m distance
                spot_growth_um = beam_deflection_mrad * 1000  # μm at 1m
                print(f"    Spot size growth @ 1m: {spot_growth_um:.1f} μm RMS")
    
    # Flatness tolerance budget analysis
    print(f"\nFlatness Tolerance Budget:")
    print(f"  Specified flatness: λ/10 @ 633nm = 63.3 nm")
    print(f"  Manufacturing tolerance: Typically λ/20 to λ/40")
    print(f"  Measurement uncertainty: ~λ/100")
    print(f"  Environmental effects: Thermal, vibration, stress")
    
    # Performance recommendations
    print(f"\nPerformance Recommendations:")
    print(f"  • For interferometry: λ/20 or better flatness")
    print(f"  • For laser beam steering: <1 mrad surface tilt")
    print(f"  • For wavefront preservation: <λ/30 OPD variation")
    print(f"  • Consider active optics for sub-nm requirements")

def main():
    print("=== Optical Flatness Tolerance Analysis ===\n")
    
    # Step 1: Set up assembly with flatness tolerances
    print("1. Setting up Optical Flat Assembly:")
    assembly, mc_analyzer = setup_flatness_analysis()
    
    print(f"   {assembly}")
    print(f"   Toleranced dimensions: {len(mc_analyzer.toleranced_dimensions)}")
    print(f"   Measurements: {len(mc_analyzer.measurements)}")
    
    # Show flatness tolerance details
    print(f"\n   Tolerance Details:")
    for dim_info in mc_analyzer.toleranced_dimensions:
        if dim_info['type'] == 'geometric' and dim_info['tolerance_type'].name == 'FLATNESS':
            flatness_nm = dim_info['tolerance_value'] * 1e6
            print(f"     {dim_info['component']}.{dim_info['datum']}: ")
            print(f"       Flatness: {flatness_nm:.1f} nm")
            print(f"       Angular equivalent: {dim_info['tolerance_value']/10*1000:.3f} mrad/10mm")
    
    for t in mc_analyzer.toleranced_dimensions:
        print(t)
        
    # print(assembly.list_mates())
    # for name,comp in assembly.components.items():
    #     print(name, comp)
    #     for dname,dat in comp.datums.items():
    #         print(dname, dat)
    #         print(dat.geo)
    
    # Step 2: Run Monte Carlo analysis
    print(f"\n2. Running Monte Carlo Analysis:")
    n_samples = 500  # Moderate sample size for this example
    sigma_factor = 3.0
    
    print(f"   Samples: {n_samples}")
    print(f"   Modeling flatness as angular deviations in surface normal")
    print(f"   Small angle approximation: sin(θ) ≈ θ")
    
    # Run the analysis
    results_df = mc_analyzer.run_monte_carlo(n_samples=n_samples, sigma_factor=sigma_factor)
    
    print(f"   Results shape: {results_df.shape}")
    
    # Step 3: Export results
    print(f"\n3. Exporting Results:")
    csv_filename = "flatness_analysis_results.csv"
    mc_analyzer.export_results(results_df, csv_filename)
    
    # Step 4: Statistical analysis
    print(f"\n4. Statistical Analysis:")
    analysis = mc_analyzer.analyze_results(results_df)
    mc_analyzer.print_analysis_summary(analysis)
    
    # Step 5: Flatness-specific analysis
    analyze_flatness_effects(results_df)
    
    # Step 6: Summary
    print(f"\n=== Flatness Analysis Complete ===")
    print(f"This analysis demonstrates:")
    print(f"• Flatness tolerance modeling as angular surface deviations")
    print(f"• Monte Carlo sampling of surface normal variations")
    print(f"• Impact on optical alignment and wavefront quality")
    print(f"• Translation of flatness specs to performance metrics")
    print(f"\nKey insights:")
    print(f"• λ/10 flatness generates measurable optical path differences")  
    print(f"• Surface tilt affects beam pointing accuracy")
    print(f"• Flatness tolerance cascades through optical system")
    print(f"• Small angle approximation valid for precision optics")

if __name__ == "__main__":
    main()