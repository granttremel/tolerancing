#!/usr/bin/env python3
"""
Example of building datums for a precision optical lens mount.

This demonstrates how the parametric datum system would be used to design
a real optical component with multiple features and tight tolerances.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tolerancing.datum import Datum, Tolerance, ToleranceType
from tolerancing.geometry import GeometryType

def main():
    print("=== Optical Lens Mount Design ===\n")
    print("Building datums for a precision lens mount with:")
    print("- Base mounting plane")
    print("- Central optical axis")
    print("- Lens seat at specified distance")
    print("- Mounting holes at precise locations")
    print("- Threaded features for adjustments\n")
    
    # Primary datums - these establish the coordinate system
    print("1. Primary Datums (establish coordinate system):")
    
    # Datum A: Base mounting plane
    base_plane = Datum.from_geometry(
        geometry_type=GeometryType.PLANE,
        origin=[0, 0, 0],
        frame=[0, 0, 1],  # Normal pointing up (+Z)
        name="A_BASE"
    )
    print(f"   {base_plane} - Base mounting surface")
    
    # Datum B: Central optical axis
    optical_axis = Datum.from_geometry(
        geometry_type=GeometryType.AXIS,
        origin=[0, 0, 0],
        frame=[0, 0, 1],  # Along +Z axis
        name="B_OPTICAL"
    )
    print(f"   {optical_axis} - Central optical axis")
    
    # Secondary datum features
    print("\n2. Secondary Datums (dimensioned from primaries):")
    
    # Lens seat plane - 50mm above base with tight tolerance
    lens_seat = base_plane.offset_distance(
        distance=50.0,
        direction=[0, 0, 1],
        target_geometry=GeometryType.PLANE,
        name="LENS_SEAT"
    )
    # Add tight dimensional tolerance for optical precision
    if lens_seat.dimension:
        lens_seat.dimension.set_tolerance(Tolerance(ToleranceType.DIMENSION, 0.01, 0.01))
    print(f"   {lens_seat} - Lens seat (50.000±0.010mm from base)")
    
    # Threaded hole for lens retainer - coaxial with optical axis, 45mm from base
    retainer_axis = optical_axis.offset_distance(
        distance=45.0,
        direction=[0, 0, 1],
        target_geometry=GeometryType.AXIS,
        name="RETAINER_THREAD"
    )
    print(f"   {retainer_axis} - Lens retainer thread axis")
    
    # Mounting holes - precision located from optical axis
    print("\n3. Mounting Hole Pattern (precise positioning):")
    
    # Four mounting holes in a square pattern
    hole_radius = 30.0  # 30mm from center
    hole_positions = [
        (hole_radius, 0),      # +X
        (0, hole_radius),      # +Y  
        (-hole_radius, 0),     # -X
        (0, -hole_radius)      # -Y
    ]
    
    mounting_holes = []
    for i, (x, y) in enumerate(hole_positions):
        hole_center = base_plane.offset_xy(
            dx=x, dy=y,
            target_geometry=GeometryType.POINT,
            name=f"HOLE_{i+1}"
        )
        # Add position tolerance for mounting precision
        hole_center.set_tolerance(Tolerance(ToleranceType.POSITION, 0.05))
        mounting_holes.append(hole_center)
        print(f"   {hole_center} - Mounting hole at ({x:+.0f}, {y:+.0f}) ⌖0.05")
    
    # Adjustment features
    print("\n4. Adjustment Features:")
    
    # Tilt adjustment points - 3 points at 120° intervals, 25mm radius
    tilt_radius = 25.0
    tilt_angles = [0, 2*np.pi/3, 4*np.pi/3]  # 0°, 120°, 240°
    
    tilt_points = []
    for i, angle in enumerate(tilt_angles):
        x = tilt_radius * np.cos(angle)
        y = tilt_radius * np.sin(angle)
        
        tilt_point = base_plane.offset_xy(
            dx=x, dy=y,
            target_geometry=GeometryType.POINT,
            name=f"TILT_{i+1}"
        )
        tilt_point.set_tolerance(Tolerance(ToleranceType.POSITION, 0.02))
        tilt_points.append(tilt_point)
        print(f"   {tilt_point} - Tilt adjustment at {np.degrees(angle):3.0f}° ⌖0.02")
    
    # Analysis of datum relationships
    print("\n5. Datum Dependency Analysis:")
    
    print("   Construction chains:")
    for datum in [lens_seat, retainer_axis, mounting_holes[0], tilt_points[0]]:
        chain = datum.get_construction_chain()
        chain_names = " → ".join([d.name for d in chain])
        print(f"     {datum.name}: {chain_names}")
    
    print("\n   Primary datums (independent):", len([d for d in [base_plane, optical_axis]]))
    print("   Derived datums (dependent):", len([lens_seat, retainer_axis] + mounting_holes + tilt_points))
    
    # Tolerance stackup preview
    print("\n6. Critical Tolerance Stackup:")
    print("   Lens position stack (base → lens seat):")
    print("     Base reference: 0.000")
    if lens_seat.dimension and lens_seat.dimension.tolerance:
        print(f"     + 50.000±{lens_seat.dimension.tolerance.pos:.3f}mm")
        print(f"     = 50.000±{lens_seat.dimension.tolerance.pos:.3f}mm total")
    
    print("\n7. Geometric Validation:")
    all_datums = [base_plane, optical_axis, lens_seat, retainer_axis] + mounting_holes + tilt_points
    
    for datum in all_datums:
        status = "✓ DEFINED" if datum.is_defined() else "✗ UNDER-DEFINED"
        print(f"   {datum.name:15} {status}")
    
    print("\n=== Lens Mount Design Complete ===")
    print("This parametric approach ensures:")
    print("• All features are properly constrained")
    print("• Tolerances are associated with dimensions")
    print("• Construction history is preserved")
    print("• Stackup analysis is possible")
    print("• Manufacturing intent is clear")

if __name__ == "__main__":
    main()