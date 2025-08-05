#!/usr/bin/env python3
"""
Complete example showing component creation, parametric datum building, and assembly mating.

This demonstrates the full workflow:
1. Create components with parametric datums
2. Add components to an assembly
3. Mate components using their datums
4. Validate the assembly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tolerancing.component import Component
from tolerancing.assembly import Assembly, MateType
from tolerancing.datum import Datum, Tolerance, ToleranceType
from tolerancing.geometry import GeometryType

def create_base_plate():
    """Create a base plate component with mounting features"""
    base = Component("BASE_PLATE")
    
    # Add primary datums
    plane_id, axis_id = base.add_primary_datums(
        plane_origin=[0, 0, 0],
        plane_normal=[0, 0, 1],  # Top surface
        axis_origin=[0, 0, 0], 
        axis_direction=[0, 0, 1]
    )
    
    # Add mounting holes using parametric datums
    hole_positions = [(50, 50), (50, -50), (-50, 50), (-50, -50)]
    
    for i, (x, y) in enumerate(hole_positions):
        hole_id = base.create_derived_datum(
            reference_id=plane_id,
            dimension_type='offset_xy',
            values=[x, y],
            target_geometry=GeometryType.POINT,
            datum_id=f"HOLE_{i+1}",
            name=f"BASE_HOLE_{i+1}"
        )
        
        # Add position tolerance to holes
        hole_datum = base.get_datum(hole_id)
        hole_datum.set_tolerance(Tolerance(ToleranceType.POSITION, 0.1))
    
    # Add a central boss (cylinder) for mating
    boss_id = base.create_derived_datum(
        reference_id=plane_id,
        dimension_type='offset_distance',
        values=[10.0],  # 10mm high
        target_geometry=GeometryType.CYLINDER,
        constraints={'direction': [0, 0, 1], 'radius': 25.0},
        datum_id="BOSS",
        name="BASE_BOSS"
    )
    
    print(f"Created {base}")
    print(f"  Datums: {base.list_datums()}")
    return base

def create_lens_mount():
    """Create a lens mount component that will mate to the base"""
    mount = Component("LENS_MOUNT")
    
    # Add primary datums - bottom face and central bore
    plane_id, axis_id = mount.add_primary_datums(
        plane_origin=[0, 0, 0],
        plane_normal=[0, 0, -1],  # Bottom face (normal pointing down)
        axis_origin=[0, 0, 0],
        axis_direction=[0, 0, 1]
    )
    
    # Create a central bore that will mate with base boss
    bore_id = mount.create_derived_datum(
        reference_id=axis_id,
        dimension_type='offset_distance', 
        values=[0.0],  # Concentric with axis
        target_geometry=GeometryType.CYLINDER,
        constraints={'direction': [0, 0, 1], 'radius': 25.1},  # Slightly larger for clearance
        datum_id="BORE",
        name="MOUNT_BORE"
    )
    
    # Create lens seat 40mm above bottom
    lens_seat_id = mount.create_derived_datum(
        reference_id=plane_id,
        dimension_type='offset_distance',
        values=[40.0],
        target_geometry=GeometryType.PLANE,
        constraints={'direction': [0, 0, 1]},
        datum_id="LENS_SEAT",
        name="LENS_SEAT_PLANE"
    )
    
    # Add tight tolerance to lens seat dimension
    lens_seat_datum = mount.get_datum(lens_seat_id)
    if lens_seat_datum.dimension:
        lens_seat_datum.dimension.set_tolerance(Tolerance(ToleranceType.DIMENSION, 0.01, 0.01))
    
    # Add adjustment points around the perimeter
    adjustment_angles = [0, np.pi*2/3, np.pi*4/3]  # 0°, 120°, 240°
    radius = 35.0
    
    for i, angle in enumerate(adjustment_angles):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        adj_id = mount.create_derived_datum(
            reference_id=plane_id,
            dimension_type='offset_xy',
            values=[x, y],
            target_geometry=GeometryType.POINT,
            datum_id=f"ADJ_{i+1}",
            name=f"ADJUSTMENT_{i+1}"
        )
    
    print(f"Created {mount}")
    print(f"  Datums: {mount.list_datums()}")
    return mount

def create_lens():
    """Create a simple lens component"""
    lens = Component("LENS")
    
    # Simple lens with two surfaces
    plane_id, axis_id = lens.add_primary_datums(
        plane_origin=[0, 0, 0],
        plane_normal=[0, 0, -1],  # Back surface
        axis_origin=[0, 0, 0],
        axis_direction=[0, 0, 1]  # Optical axis
    )
    
    # Front surface
    front_id = lens.create_derived_datum(
        reference_id=plane_id,
        dimension_type='offset_distance',
        values=[5.0],  # 5mm thick
        target_geometry=GeometryType.PLANE,
        constraints={'direction': [0, 0, 1]},
        datum_id="FRONT",
        name="LENS_FRONT"
    )
    
    print(f"Created {lens}")
    print(f"  Datums: {lens.list_datums()}")
    return lens

def main():
    print("=== Component and Assembly Mating Example ===\n")
    
    # Step 1: Create components
    print("1. Creating Components:")
    base_plate = create_base_plate()
    lens_mount = create_lens_mount()
    lens = create_lens()
    
    # Step 2: Create assembly and add components
    print("\n2. Creating Assembly:")
    assembly = Assembly("OPTICAL_SYSTEM")
    
    # Add base plate as ground component (fixed reference)
    assembly.add_component(base_plate, ground=True)
    
    # Add other components (initially floating)
    assembly.add_component(lens_mount)
    assembly.add_component(lens)
    
    print(f"Created {assembly}")
    print(f"  Components: {assembly.list_components()}")
    
    # Step 3: Mate components
    print("\n3. Mating Components:")
    
    # Mate lens mount to base plate
    # Boss-bore concentric mate
    mate1 = assembly.mate_components(
        "BASE_PLATE", "BOSS",      # Base boss
        "LENS_MOUNT", "BORE",      # Mount bore
        MateType.CONCENTRIC
    )
    
    # Bottom face coincident mate
    mate2 = assembly.mate_components(
        "BASE_PLATE", "A",         # Base top surface  
        "LENS_MOUNT", "A",         # Mount bottom surface
        MateType.COINCIDENT
    )
    
    # Mate lens to lens mount
    mate3 = assembly.mate_components(
        "LENS_MOUNT", "LENS_SEAT", # Mount lens seat
        "LENS", "A",               # Lens back surface
        MateType.COINCIDENT
    )
    
    # Concentric optical axes
    mate4 = assembly.mate_components(
        "LENS_MOUNT", "B",         # Mount optical axis
        "LENS", "B",               # Lens optical axis  
        MateType.CONCENTRIC
    )
    
    print(f"  Mates created: {len(assembly.mates)}")
    print(f"  Mate list: {assembly.list_mates()}")
    
    # Step 4: Validate assembly
    print("\n4. Assembly Validation:")
    
    print(f"  Assembly status: {assembly}")
    print(f"  Fully constrained: {assembly.is_fully_constrained()}")
    
    issues = assembly.validate_assembly()
    if any(issues.values()):
        print("  Issues found:")
        for issue_type, problems in issues.items():
            if problems:
                print(f"    {issue_type}: {problems}")
    else:
        print("  ✓ No issues found - assembly is valid!")
    
    # Step 5: Show datum transfer map (for tolerance analysis)
    print("\n5. Datum Transfer Map:")
    transfer_map = assembly.get_datum_transfer_map()
    
    print("  Datum connections (for tolerance stackup):")
    for datum, connections in transfer_map.items():
        print(f"    {datum} ↔ {connections}")
    
    # Step 6: Component positioning status
    print("\n6. Component Positioning:")
    for name, comp in assembly.components.items():
        status = "POSITIONED" if comp.is_positioned else "FLOATING"
        ground_status = " (GROUND)" if comp == assembly.ground_component else ""
        print(f"   {name}: {status}{ground_status}")
        if comp.is_positioned and hasattr(comp, 'position'):
            pos = comp.position
            print(f"     Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
    
    # Step 7: Show critical dimension chain
    print("\n7. Critical Dimension Analysis:")
    print("   Lens position stackup (base → lens):")
    print("     BASE_PLATE.A (reference) → LENS_MOUNT.LENS_SEAT → LENS.A")
    
    # Get the lens seat dimension with tolerance
    lens_seat_datum = lens_mount.get_datum("LENS_SEAT")
    if lens_seat_datum.dimension and lens_seat_datum.dimension.tolerance:
        tol = lens_seat_datum.dimension.tolerance
        print(f"     Mount height: 40.000±{tol.pos:.3f}mm")
        print(f"     Total lens position: 40.000±{tol.pos:.3f}mm from base")
    
    print("\n=== Complete Assembly Example Finished ===")
    print("This demonstrates:")
    print("• Parametric datum creation in components")
    print("• Assembly of multiple components")
    print("• Geometric mate validation")
    print("• Automatic component positioning")
    print("• Datum transfer mapping for tolerance analysis")
    print("• Full assembly validation")

if __name__ == "__main__":
    main()