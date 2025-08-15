#!/usr/bin/env python3
"""
Comprehensive demonstration of the Component display capabilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tolerancing import Component, GeometryType
from tolerancing.datum import DOF, Tolerance, ToleranceType


def main():
    """Demonstrate the component display features."""
    
    print("\n" + "="*70)
    print(" COMPONENT DISPLAY DEMONSTRATION ")
    print("="*70)
    
    # Create a simple component
    comp = Component("SimpleBlock")
    
    print("\n1. SIMPLE COMPONENT:")
    print("-" * 40)
    
    # Add a few geometries
    comp.add_plane(reference_id="PlaneA", offset=10, datum_label="B")
    comp.add_axis(reference_id="PlaneA", origin=[5, 0, 0], datum_label="C")
    comp.add_cylinder(reference_id="PlaneB", origin=[0, 0, 5],frame = [0,1,1], radius=2.5)
    comp.add_point(reference_id="AxisA", position=[0, 0, 10])
    
    # Set some tolerances
    comp.set_tolerance("PlaneA", ToleranceType.FLATNESS, 0.002)
    comp.set_tolerance("PlaneB", ToleranceType.PARALLELISM, 0.001, dof=DOF.RX)
    comp.set_tolerance("CylinderA", ToleranceType.CYLINDRICITY, 0.005)
    
    # Display the tree
    comp.display()
    
    print("\n2. GEOMETRY ACCESS EXAMPLES:")
    print("-" * 40)
    print(f"comp.PlaneA:    {comp.PlaneA}")
    print(f"comp.PlaneB:    {comp.PlaneB}")
    print(f"comp.AxisA:     {comp.AxisA}")
    print(f"comp.CylinderA: {comp.CylinderA}")
    print(f"comp.PointA:    {comp.PointA}")
    
    print("\n3. DATUM ACCESS (by label):")
    print("-" * 40)
    print(f"comp.A (Datum A): {comp.A}")
    print(f"comp.B (Datum B): {comp.B}")
    print(f"comp.C (Datum C): {comp.C}")
    
    print("\n4. RELATIONSHIP QUERIES:")
    print("-" * 40)
    print(f"Parent of CylinderA: {comp.get_parent('CylinderA')}")
    print(f"Children of PlaneA:  {comp.get_children('PlaneA')}")
    print(f"Children of PlaneB:  {comp.get_children('PlaneB')}")
    print(f"Ancestry of PointA:  {comp._get_ancestry('PointA')}")
    
    print("\n5. TOLERANCE SUMMARY:")
    print("-" * 40)
    summary = comp.get_tolerance_summary("CylinderA")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print(" END OF DEMONSTRATION ")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())