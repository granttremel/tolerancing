#!/usr/bin/env python3
"""
Demonstrate the component tree display functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tolerancing import Component, GeometryType
from tolerancing.datum import DOF, Tolerance, ToleranceType


def create_sample_component():
    """Create a sample component with various geometries and tolerances."""
    
    # Create component
    comp = Component("OpticalMount")
    
    # Add base plane with tolerance
    comp.set_tolerance("PlaneA", ToleranceType.FLATNESS, 0.005)
    
    # Add mounting holes
    hole1 = comp.add_cylinder(
        reference_id="PlaneA",
        origin=[20, 20, 0],
        radius=3.0,
        datum_label="B"
    )
    comp.set_tolerance(hole1, ToleranceType.CYLINDRICITY, 0.01)
    comp.set_tolerance(hole1, ToleranceType.POSITION, 0.05, dof=DOF.DX)
    comp.set_tolerance(hole1, ToleranceType.POSITION, 0.05, dof=DOF.DY)
    
    hole2 = comp.add_cylinder(
        reference_id="PlaneA",
        origin=[-20, 20, 0],
        radius=3.0
    )
    comp.set_tolerance(hole2, ToleranceType.POSITION, 0.05, dof=DOF.DX)
    comp.set_tolerance(hole2, ToleranceType.POSITION, 0.05, dof=DOF.DY)
    
    # Add reference surface
    ref_plane = comp.add_plane(
        reference_id="PlaneA",
        offset=10,
        datum_label="C"
    )
    comp.set_tolerance(ref_plane, ToleranceType.PARALLELISM, 0.002, dof=DOF.RX)
    
    # Add optical axis
    optical_axis = comp.add_axis(
        reference_id=ref_plane,
        origin=[0, 0, 5],
        datum_label="D"
    )
    comp.set_tolerance(optical_axis, ToleranceType.PERPENDICULARITY, 0.001, dof=DOF.RX)
    comp.set_tolerance(optical_axis, ToleranceType.PERPENDICULARITY, 0.001, dof=DOF.RY)
    
    # Add measurement points
    point1 = comp.add_point(reference_id=optical_axis, position=[0, 0, 20])
    point2 = comp.add_point(reference_id=optical_axis, position=[0, 0, 40])
    
    # Add another branch from hole1
    feature = comp.add_plane(reference_id=hole1, offset=5)
    
    return comp


def main():
    """Run demonstration."""
    print("\n" + "="*60)
    print("COMPONENT TREE DISPLAY DEMONSTRATION")
    print("="*60)
    
    # Create sample component
    comp = create_sample_component()
    
    # Show basic info
    print(f"\nComponent representation: {comp}")
    
    # Show tree without details
    print("\n\n1. BASIC TREE (no details):")
    comp.print_tree(show_tolerances=False, show_dimensions=False)
    
    # Show tree with dimensions
    print("\n\n2. TREE WITH DIMENSIONS:")
    comp.print_tree(show_tolerances=False, show_dimensions=True)
    
    # Show tree with tolerances
    print("\n\n3. TREE WITH TOLERANCES:")
    comp.print_tree(show_tolerances=True, show_dimensions=False)
    
    # Show full tree
    print("\n\n4. FULL TREE (all details):")
    comp.display()
    
    # Test attribute access
    print("\n\n5. ATTRIBUTE ACCESS:")
    print(f"comp.PlaneA: {comp.PlaneA}")
    print(f"comp.B (Datum B): {comp.B}")
    print(f"comp.CylinderA: {comp.CylinderA}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())