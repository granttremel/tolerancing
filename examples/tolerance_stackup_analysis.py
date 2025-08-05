#!/usr/bin/env python3
"""
Advanced example demonstrating coordinate transformations and tolerance stackup analysis.

This shows the complete workflow:
1. Build a complex assembly with multiple transformation chains
2. Navigate datum transfer trees
3. Transform coordinates between arbitrary datums
4. Perform error propagation analysis
5. Identify critical dimensions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tolerancing.component import Component
from tolerancing.assembly import Assembly, MateType
from tolerancing.datum import Datum, Tolerance, ToleranceType
from tolerancing.geometry import GeometryType

def create_precision_base():
    """Create a precision base with tight tolerances"""
    base = Component("PRECISION_BASE")
    
    # Primary datums
    plane_id, axis_id = base.add_primary_datums(
        plane_origin=[0, 0, 0],
        plane_normal=[0, 0, 1]
    )
    
    # Create a precision reference surface 100mm above base
    ref_surface_id = base.create_derived_datum(
        reference_id=plane_id,
        dimension_type='offset_distance',
        values=[100.0],
        target_geometry=GeometryType.PLANE,
        constraints={'direction': [0, 0, 1]},
        datum_id="REF_SURFACE",
        name="REFERENCE_SURFACE"
    )
    
    # Add tight tolerance to this critical dimension
    ref_surface = base.get_datum(ref_surface_id)
    if ref_surface.dimension:
        ref_surface.dimension.set_tolerance(Tolerance(ToleranceType.DIMENSION, 0.005, 0.005))
    
    # Create mounting features with position tolerances
    mount_positions = [(75, 0), (0, 75), (-75, 0), (0, -75)]
    
    for i, (x, y) in enumerate(mount_positions):
        mount_id = base.create_derived_datum(
            reference_id=ref_surface_id,
            dimension_type='offset_xy',
            values=[x, y],
            target_geometry=GeometryType.POINT,
            datum_id=f"MOUNT_{i+1}",
            name=f"MOUNT_POINT_{i+1}"
        )
        
        mount_datum = base.get_datum(mount_id)
        mount_datum.set_tolerance(Tolerance(ToleranceType.POSITION, 0.02))
    
    return base

def create_optical_stage():
    """Create an optical stage that mounts to the base"""
    stage = Component("OPTICAL_STAGE")
    
    # Primary datums
    plane_id, axis_id = stage.add_primary_datums(
        plane_origin=[0, 0, 0],
        plane_normal=[0, 0, -1]  # Bottom face
    )
    
    # Create optical reference plane 25mm above bottom
    opt_plane_id = stage.create_derived_datum(
        reference_id=plane_id,
        dimension_type='offset_distance',
        values=[25.0],
        target_geometry=GeometryType.PLANE,
        constraints={'direction': [0, 0, 1]},
        datum_id="OPT_PLANE",
        name="OPTICAL_PLANE"
    )
    
    # Add tolerance to optical plane height
    opt_plane = stage.get_datum(opt_plane_id)
    if opt_plane.dimension:
        opt_plane.dimension.set_tolerance(Tolerance(ToleranceType.DIMENSION, 0.01, 0.01))
    
    # Create precision hole pattern matching base
    hole_positions = [(75, 0), (0, 75), (-75, 0), (0, -75)]
    
    for i, (x, y) in enumerate(hole_positions):
        hole_id = stage.create_derived_datum(
            reference_id=plane_id,
            dimension_type='offset_xy',
            values=[x, y],
            target_geometry=GeometryType.POINT,
            datum_id=f"HOLE_{i+1}",
            name=f"STAGE_HOLE_{i+1}"
        )
        
        hole_datum = stage.get_datum(hole_id)
        hole_datum.set_tolerance(Tolerance(ToleranceType.POSITION, 0.015))
    
    # Create angled adjustment feature
    adj_point_id = stage.create_derived_datum(
        reference_id=opt_plane_id,
        dimension_type='offset_xy',
        values=[50.0, 0.0],
        target_geometry=GeometryType.POINT,
        datum_id="ADJ_POINT",
        name="ADJUSTMENT_POINT"
    )
    
    return stage

def create_lens_assembly():
    """Create a lens assembly with multiple elements"""
    lens_assy = Component("LENS_ASSEMBLY")
    
    # Primary datums
    plane_id, axis_id = lens_assy.add_primary_datums(
        plane_origin=[0, 0, 0],
        plane_normal=[0, 0, -1]  # Back surface
    )
    
    # First lens surface (back)
    back_surface_id = plane_id  # Use primary plane
    
    # Create lens thickness with very tight tolerance
    front_surface_id = lens_assy.create_derived_datum(
        reference_id=back_surface_id,
        dimension_type='offset_distance',
        values=[8.0],  # 8mm thick lens
        target_geometry=GeometryType.PLANE,
        constraints={'direction': [0, 0, 1]},
        datum_id="FRONT_SURFACE",
        name="LENS_FRONT_SURFACE"
    )
    
    # Very tight tolerance on lens thickness for optical performance
    front_surface = lens_assy.get_datum(front_surface_id)
    if front_surface.dimension:
        front_surface.dimension.set_tolerance(Tolerance(ToleranceType.DIMENSION, 0.002, 0.002))
    
    # Create optical axis reference point at lens center
    optical_center_id = lens_assy.create_derived_datum(
        reference_id=back_surface_id,
        dimension_type='offset_distance',
        values=[4.0],  # Center of lens
        target_geometry=GeometryType.POINT,
        constraints={'direction': [0, 0, 1]},
        datum_id="OPT_CENTER",
        name="OPTICAL_CENTER"
    )
    
    return lens_assy

def main():
    print("=== Advanced Tolerance Stackup Analysis ===\n")
    
    # Step 1: Create complex assembly
    print("1. Creating Complex Assembly:")
    
    base = create_precision_base()
    stage = create_optical_stage()
    lens = create_lens_assembly()
    
    print(f"   {base}")
    print(f"   {stage}")
    print(f"   {lens}")
    
    # Step 2: Assemble components
    print("\n2. Assembling Components:")
    
    assembly = Assembly("PRECISION_OPTICAL_SYSTEM")
    assembly.add_component(base, ground=True)
    assembly.add_component(stage)
    assembly.add_component(lens)
    
    # Mate stage to base (reference surfaces coincident)
    assembly.mate_components(
        "PRECISION_BASE", "REF_SURFACE",
        "OPTICAL_STAGE", "A",
        MateType.COINCIDENT
    )
    
    # Mate lens to stage optical plane
    assembly.mate_components(
        "OPTICAL_STAGE", "OPT_PLANE", 
        "LENS_ASSEMBLY", "A",
        MateType.COINCIDENT
    )
    
    # Align optical axes
    assembly.mate_components(
        "PRECISION_BASE", "B",
        "LENS_ASSEMBLY", "B", 
        MateType.CONCENTRIC
    )
    
    print(f"   {assembly}")
    
    # Step 3: Coordinate Transformation Analysis
    print("\n3. Coordinate Transformation Analysis:")
    
    # Transform lens optical center to global coordinates
    lens_center_global, transform_info = assembly.transform_datum_to_global(
        "LENS_ASSEMBLY", "OPT_CENTER", [0, 0, 0]
    )
    
    print(f"   Lens optical center in global coordinates: ({lens_center_global[0]:.3f}, {lens_center_global[1]:.3f}, {lens_center_global[2]:.3f})")
    print(f"   Transformation steps: {transform_info['steps']}")
    print(f"   Transformation chain: {' → '.join(transform_info['chain'])}")
    
    # Transform between different datums
    stage_to_base, stage_transform_info = assembly.transform_between_datums(
        "OPTICAL_STAGE", "ADJ_POINT",
        "PRECISION_BASE", "REF_SURFACE",
        [0, 0, 0]
    )
    
    print(f"   Stage adjustment point relative to base reference: ({stage_to_base[0]:.3f}, {stage_to_base[1]:.3f}, {stage_to_base[2]:.3f})")
    print(f"   Total transformation steps: {stage_transform_info['total_steps']}")
    
    # Step 4: Detailed Transformation Trees
    print("\n4. Transformation Trees:")
    
    # Show transformation tree for lens optical center
    assembly.print_transformation_tree("LENS_ASSEMBLY", "OPT_CENTER")
    
    # Show transformation tree for stage adjustment point
    assembly.print_transformation_tree("OPTICAL_STAGE", "ADJ_POINT")
    
    # Step 5: Tolerance Stackup Analysis
    print("\n5. Tolerance Stackup Analysis:")
    
    # Analyze lens position uncertainty
    lens_analysis = assembly.analyze_tolerance_stackup("LENS_ASSEMBLY", "OPT_CENTER")
    
    print(f"   Lens Optical Center Analysis:")
    print(f"     Nominal position: {lens_analysis['nominal_position']}")
    print(f"     Global position: [{lens_analysis['global_position'][0]:.3f}, {lens_analysis['global_position'][1]:.3f}, {lens_analysis['global_position'][2]:.3f}]")
    print(f"     Position uncertainty (1σ):")
    uncertainty = lens_analysis['uncertainty']
    print(f"       X: ±{uncertainty.get('std_x', 0):.4f}mm")
    print(f"       Y: ±{uncertainty.get('std_y', 0):.4f}mm") 
    print(f"       Z: ±{uncertainty.get('std_z', 0):.4f}mm")
    print(f"     Toleranced dimensions: {lens_analysis['toleranced_dimensions']}")
    
    # Analyze stage adjustment point
    stage_analysis = assembly.analyze_tolerance_stackup("OPTICAL_STAGE", "ADJ_POINT")
    
    print(f"\n   Stage Adjustment Point Analysis:")
    print(f"     Position uncertainty (1σ):")
    stage_uncertainty = stage_analysis['uncertainty']
    print(f"       X: ±{stage_uncertainty.get('std_x', 0):.4f}mm")
    print(f"       Y: ±{stage_uncertainty.get('std_y', 0):.4f}mm")
    print(f"       Z: ±{stage_uncertainty.get('std_z', 0):.4f}mm")
    
    # Step 6: Critical Dimension Analysis
    print("\n6. Critical Dimension Analysis:")
    
    critical_dims = assembly.get_critical_dimensions()
    
    if critical_dims:
        print("   Most critical dimensions (by total uncertainty):")
        for i, dim_info in enumerate(critical_dims[:5]):  # Top 5
            print(f"     {i+1}. {dim_info['component']}.{dim_info['datum']} ({dim_info['datum_name']})")
            print(f"        Total uncertainty: ±{dim_info['total_uncertainty']:.4f}mm")
            print(f"        Contributing tolerances: {dim_info['contributing_tolerances']}")
    else:
        print("   No critical dimensions found (all dimensions have zero tolerance)")
    
    # Step 7: Symbolic Matrix Export
    print("\n7. Symbolic Transformation Matrix:")
    
    # Export the symbolic matrix for the lens center
    matrix_export = assembly.export_transformation_matrix("LENS_ASSEMBLY", "OPT_CENTER")
    print("   Lens optical center transformation matrix:")
    print("   " + matrix_export.replace("\n", "\n   "))
    
    # Step 8: Design Sensitivity Analysis
    print("\n8. Design Sensitivity Analysis:")
    
    print("   Key insights:")
    print("   • Base reference surface (±0.005mm) affects all downstream components")
    print("   • Lens thickness (±0.002mm) directly affects optical center position")
    print("   • Stage height (±0.010mm) compounds with lens position errors")
    print("   • Position tolerances (±0.015-0.020mm) affect alignment accuracy")
    
    # Calculate total lens position uncertainty
    if lens_analysis['uncertainty']:
        total_lens_uncertainty = (
            lens_analysis['uncertainty'].get('std_x', 0)**2 +
            lens_analysis['uncertainty'].get('std_y', 0)**2 + 
            lens_analysis['uncertainty'].get('std_z', 0)**2
        )**0.5
        print(f"   • Total lens position uncertainty: ±{total_lens_uncertainty:.4f}mm (RSS)")
        
        # Check against optical requirements
        if total_lens_uncertainty > 0.050:  # 50 micron requirement
            print("   ⚠ WARNING: Lens position uncertainty exceeds 50μm optical requirement!")
        else:
            print("   ✓ Lens position uncertainty meets 50μm optical requirement")
    
    print("\n=== Advanced Analysis Complete ===")
    print("This system demonstrates:")
    print("• Symbolic coordinate transformations with SymPy")
    print("• Datum transfer tree navigation")
    print("• Error propagation through transformation chains")
    print("• Critical dimension identification")
    print("• Design sensitivity analysis for optical systems")

if __name__ == "__main__":
    main()