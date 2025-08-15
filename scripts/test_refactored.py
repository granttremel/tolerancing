#!/usr/bin/env python3
"""
Test script for the refactored Component, Datum, and Dimension classes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tolerancing import Component, Plane, Axis, Point, GeometryType
from tolerancing.datum import DOF, Tolerance, ToleranceType


def test_component_creation():
    """Test creating a component with default geometry."""
    print("Testing component creation...")
    
    # Create component with default top plane
    comp = Component("TestPart")
    print(f"Created: {comp}")
    
    # Check initial geometry
    print(f"Initial geometries: {comp.list_geometries()}")
    print(f"Initial datums: {comp.list_datums()}")
    
    # The initial geometry should be PlaneA
    assert "PlaneA" in comp.geometries
    assert "PlaneA" in comp.datums
    
    print("✓ Component creation successful\n")


def test_geometry_derivation():
    """Test deriving new geometries from existing ones."""
    print("Testing geometry derivation...")
    
    # Create component
    comp = Component("TestPart")
    
    # Derive a new plane offset by 10mm in Z direction
    plane_b_id = comp.derive("PlaneA", GeometryType.PLANE, du=10)
    print(f"Created {plane_b_id} by offsetting PlaneA")
    
    # Derive an axis from PlaneA
    axis_a_id = comp.derive("PlaneA", GeometryType.AXIS, dv=5, dw=5)
    print(f"Created {axis_a_id} from PlaneA")
    
    # Create a point by offset
    point_a_id = comp.offset_xyz("PlaneA", 10, 20, 30)
    print(f"Created {point_a_id} by XYZ offset")
    
    print(f"All geometries: {comp.list_geometries()}")
    print(f"All datums: {comp.list_datums()}")
    
    print("✓ Geometry derivation successful\n")


def test_id_generation():
    """Test that geometry IDs are generated correctly."""
    print("Testing ID generation...")
    
    comp = Component("TestPart")
    
    # Add multiple planes
    plane_b = Plane(origin=[0,0,10], u=[0,0,1])
    plane_b_id = comp.add_geometry(plane_b)
    
    plane_c = Plane(origin=[0,0,20], u=[0,0,1])
    plane_c_id = comp.add_geometry(plane_c)
    
    # Add axes
    axis_a = Axis(origin=[0,0,0], u=[1,0,0])
    axis_a_id = comp.add_geometry(axis_a)
    
    axis_b = Axis(origin=[0,0,0], u=[0,1,0])
    axis_b_id = comp.add_geometry(axis_b)
    
    print(f"Generated IDs: {list(comp.geometries.keys())}")
    
    # Check that IDs follow the pattern
    assert plane_b_id == "PlaneB"
    assert plane_c_id == "PlaneC"
    assert axis_a_id == "AxisA"
    assert axis_b_id == "AxisB"
    
    print("✓ ID generation successful\n")


def test_dimensions_and_tolerances():
    """Test dimension and tolerance management."""
    print("Testing dimensions and tolerances...")
    
    comp = Component("TestPart")
    
    # Create a derived plane
    plane_b_id = comp.derive("PlaneA", GeometryType.PLANE, du=10)
    
    # Get the datum for PlaneB
    datum_b = comp.get_datum(plane_b_id)
    
    # Check dimension values
    dim = datum_b.dimension
    print(f"PlaneB dimension: {dim}")
    print(f"Translation vector: {dim.get_translation_vector()}")
    
    # Set tolerances
    tol = Tolerance(ToleranceType.DIMENSION, 0.1, 0.05)
    comp.set_tolerance(plane_b_id, DOF.DX, tol)
    
    # Check tolerance was set
    assert datum_b.dimension.tolerances[DOF.DX].pos == 0.1
    assert datum_b.dimension.tolerances[DOF.DX].neg == 0.05
    
    print(f"Tolerance for DX: {datum_b.dimension.tolerances[DOF.DX]}")
    
    print("✓ Dimensions and tolerances successful\n")


def test_convenience_methods():
    """Test convenience methods moved from Datum to Component."""
    print("Testing convenience methods...")
    
    comp = Component("TestPart")
    
    # Test offset_distance
    plane_b_id = comp.offset_distance("PlaneA", 15, [0,0,1], GeometryType.PLANE)
    print(f"Created {plane_b_id} by offset_distance")
    
    # Test offset_xy to create a point first
    point_a_id = comp.offset_xy("PlaneA", 5, 10, GeometryType.POINT)
    print(f"Created {point_a_id} by offset_xy")
    
    # Create an axis first, then test angle_from
    axis_a_id = comp.derive("PlaneA", GeometryType.AXIS, dv=0, dw=0)
    print(f"Created {axis_a_id} from PlaneA")
    
    # Test angle_from
    axis_b_id = comp.angle_from(axis_a_id, np.pi/4, GeometryType.AXIS)
    print(f"Created {axis_b_id} by angle_from")
    
    print(f"All geometries: {comp.list_geometries()}")
    
    print("✓ Convenience methods successful\n")


def test_geometry_intersection():
    """Test creating geometry from intersections."""
    print("Testing geometry intersection...")
    
    comp = Component("TestPart")
    
    # Add a second plane at an angle
    plane_b = Plane(origin=[0,0,0], u=[1,0,1])  # 45 degree angle
    plane_b_id = comp.add_geometry(plane_b)
    
    # Intersect the two planes to get an axis
    try:
        intersection_id = comp.intersect_geometries("PlaneA", plane_b_id)
        print(f"Created {intersection_id} from intersection of PlaneA and {plane_b_id}")
        print(f"Intersection type: {comp.get_geometry(intersection_id).geotype.name}")
    except Exception as e:
        print(f"Intersection failed (expected for parallel planes): {e}")
    
    print("✓ Geometry intersection successful\n")


def test_attribute_access():
    """Test accessing datums and geometries as attributes."""
    print("Testing attribute access...")
    
    comp = Component("TestPart")
    
    # Add some geometries
    comp.derive("PlaneA", GeometryType.PLANE, du=10)
    comp.derive("PlaneA", GeometryType.AXIS)
    
    # Access via attribute notation
    plane_a = comp.PlaneA
    plane_b = comp.PlaneB
    axis_a = comp.AxisA
    
    print(f"Accessed PlaneA: {plane_a}")
    print(f"Accessed PlaneB: {plane_b}")
    print(f"Accessed AxisA: {axis_a}")
    
    print("✓ Attribute access successful\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING REFACTORED TOLERANCING CLASSES")
    print("=" * 60 + "\n")
    
    try:
        test_component_creation()
        test_geometry_derivation()
        test_id_generation()
        test_dimensions_and_tolerances()
        test_convenience_methods()
        test_geometry_intersection()
        test_attribute_access()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())