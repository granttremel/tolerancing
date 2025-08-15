#!/usr/bin/env python3
"""
Test script for the refactored Component class with enhanced relationship tracking.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tolerancing import Component, Plane, Axis, Cylinder, Point, GeometryType
from tolerancing.datum import DOF, Tolerance, ToleranceType


def test_component_initialization():
    """Test component creation and initial setup."""
    print("Testing component initialization...")
    
    # Create component with default geometry
    comp = Component("TestPart")
    
    # Check initial state
    assert len(comp.geometries) == 1
    assert len(comp.datums) == 1  # Primary datum should be created
    assert comp.primary_datum_id == "PlaneA"
    
    # Check datum mapping
    assert "A" in comp.datums
    assert comp.datums["A"] == "PlaneA"
    
    print(f"  Created: {comp}")
    print(f"  Geometries: {comp.list_geometries()}")
    print(f"  Datums: {comp.list_datums()}")
    print("✓ Component initialization successful\n")


def test_geometry_relationships():
    """Test parent-child relationship tracking."""
    print("Testing geometry relationships...")
    
    comp = Component("TestPart")
    
    # Add child geometries
    plane_b_id = comp.add_plane(reference_id="PlaneA", offset=10, datum_label="B")
    axis_a_id = comp.add_axis(reference_id="PlaneA", origin=[5, 5, 0])
    cyl_a_id = comp.add_cylinder(reference_id="PlaneA", origin=[10, 10, 0], radius=3)
    
    # Check parent-child relationships
    assert comp.get_parent(plane_b_id) == "PlaneA"
    assert comp.get_parent(axis_a_id) == "PlaneA"
    assert comp.get_parent(cyl_a_id) == "PlaneA"
    
    # Check children list
    children = comp.get_children("PlaneA")
    assert plane_b_id in children
    assert axis_a_id in children
    assert cyl_a_id in children
    
    # Check ancestry
    point_a_id = comp.add_point(reference_id=cyl_a_id, position=[0, 0, 5])
    ancestors = comp._get_ancestry(point_a_id)
    assert cyl_a_id in ancestors
    assert "PlaneA" in ancestors
    
    print(f"  PlaneA children: {comp.get_children('PlaneA')}")
    print(f"  {point_a_id} ancestors: {ancestors}")
    print("✓ Relationship tracking successful\n")


def test_datum_management():
    """Test datum label assignment and retrieval."""
    print("Testing datum management...")
    
    comp = Component("TestPart")
    
    # Add geometries with explicit datum labels
    plane_b_id = comp.add_plane(reference_id="PlaneA", offset=10, datum_label="B")
    axis_a_id = comp.add_axis(reference_id="PlaneA", datum_label="C")
    
    # Check datum labels
    assert comp.get_datum_label("PlaneA") == "A"
    assert comp.get_datum_label(plane_b_id) == "B"
    assert comp.get_datum_label(axis_a_id) == "C"
    
    # Check datum geometry retrieval
    datum_a_geo = comp.get_datum_geometry("A")
    assert datum_a_geo.geotype == GeometryType.PLANE
    
    datum_b_geo = comp.get_datum_geometry("B")
    assert datum_b_geo.geotype == GeometryType.PLANE
    
    print(f"  Datum mapping: {comp.list_datums()}")
    print("✓ Datum management successful\n")


def test_convenience_methods():
    """Test convenience methods for building components."""
    print("Testing convenience methods...")
    
    comp = Component("TestPart")
    
    # Test add_plane with various options
    plane_b = comp.add_plane(reference_id="PlaneA", offset=10)
    plane_c = comp.add_plane(origin=[0, 0, 20], normal=[0, 0, 1])  # Independent plane
    
    # Test add_axis
    axis_a = comp.add_axis(reference_id="PlaneA", origin=[10, 0, 0])
    axis_b = comp.add_axis(origin=[0, 0, 0], direction=[1, 0, 0])  # Independent axis
    
    # Test add_cylinder
    cyl_a = comp.add_cylinder(reference_id="PlaneA", origin=[20, 20, 0], radius=5)
    
    # Test add_point
    point_a = comp.add_point(reference_id=cyl_a, position=[0, 0, 10])
    
    print(f"  Created geometries: {list(comp.geometries.keys())}")
    print(f"  Geometry types: {comp.list_geometries()}")
    print("✓ Convenience methods successful\n")


def test_dimension_tracking():
    """Test dimension parameter tracking."""
    print("Testing dimension tracking...")
    
    comp = Component("TestPart")
    
    # Add geometry with explicit dimension parameters
    plane_b = comp.add_plane(reference_id="PlaneA", offset=10, datum_label="B")
    axis_a = comp.add_axis(reference_id="PlaneA", origin=[5, 10, 15])
    
    # Check dimension values
    plane_b_dim = comp.dimensions[plane_b]
    assert plane_b_dim.values[DOF.DX] == 10  # offset maps to dx for planes
    
    axis_a_dim = comp.dimensions[axis_a]
    trans = axis_a_dim.get_translation_vector()
    assert np.allclose(trans, [5, 10, 15])
    
    print(f"  PlaneB dimension: {plane_b_dim}")
    print(f"  AxisA translation: {trans}")
    print("✓ Dimension tracking successful\n")


def test_tolerance_management():
    """Test tolerance setting and form error initialization."""
    print("Testing tolerance management...")
    
    comp = Component("TestPart")
    
    # Add a plane with tolerances
    plane_b = comp.add_plane(reference_id="PlaneA", offset=10, datum_label="B")
    
    # Set form tolerance (flatness)
    comp.set_tolerance(plane_b, ToleranceType.FLATNESS, 0.01)
    
    # Set position tolerance
    comp.set_tolerance(plane_b, ToleranceType.POSITION, 0.05, dof=DOF.DX)
    
    # Check tolerances were set
    tol_set = comp.tolerances[plane_b]
    assert ToleranceType.FLATNESS in tol_set.form_tolerances
    
    # Check form error was initialized
    assert comp.form_errors[plane_b] is not None
    assert comp.form_errors[plane_b]['tolerance'] == 0.01
    assert comp.form_errors[plane_b]['std_dev'] == 0.01 / 3
    
    print(f"  Form error model: {comp.form_errors[plane_b]}")
    print("✓ Tolerance management successful\n")


def test_geometry_intersection():
    """Test intersection geometry creation."""
    print("Testing geometry intersection...")
    
    comp = Component("TestPart")
    
    # Create two intersecting planes
    plane_b = comp.add_plane(origin=[0, 0, 0], normal=[1, 0, 1])  # 45-degree plane
    
    # Create intersection (should be an axis)
    intersection = comp.intersect_geometries("PlaneA", plane_b)
    
    # Check intersection was created correctly
    inter_geo = comp.get_geometry(intersection)
    assert inter_geo.geotype == GeometryType.AXIS
    
    # Check it has PlaneA as parent
    assert comp.get_parent(intersection) == "PlaneA"
    
    print(f"  Created intersection: {intersection} (type: {inter_geo.geotype.name})")
    print("✓ Geometry intersection successful\n")


def test_attribute_access():
    """Test accessing geometries via attribute notation."""
    print("Testing attribute access...")
    
    comp = Component("TestPart")
    
    # Add some geometries with datum labels
    comp.add_plane(reference_id="PlaneA", offset=10, datum_label="B")
    comp.add_axis(reference_id="PlaneA", datum_label="C")
    
    # Access by geometry ID
    plane_a = comp.PlaneA
    assert plane_a.geotype == GeometryType.PLANE
    
    plane_b = comp.PlaneB
    assert plane_b.geotype == GeometryType.PLANE
    
    # Access by datum label
    datum_a = comp.A
    assert datum_a.geotype == GeometryType.PLANE
    
    datum_b = comp.B
    assert datum_b.geotype == GeometryType.PLANE
    
    print(f"  Accessed PlaneA: {plane_a}")
    print(f"  Accessed datum B: {datum_b}")
    print("✓ Attribute access successful\n")


def test_tolerance_summary():
    """Test getting tolerance summary for a geometry."""
    print("Testing tolerance summary...")
    
    comp = Component("TestPart")
    
    # Add geometry with various tolerances
    cyl_a = comp.add_cylinder(reference_id="PlaneA", origin=[10, 10, 0], 
                             radius=3, datum_label="B")
    
    # Set tolerances
    comp.set_tolerance(cyl_a, ToleranceType.CYLINDRICITY, 0.005)
    comp.set_tolerance(cyl_a, ToleranceType.POSITION, 0.1, dof=DOF.DX)
    comp.set_tolerance(cyl_a, ToleranceType.POSITION, 0.1, dof=DOF.DY)
    
    # Get summary
    summary = comp.get_tolerance_summary(cyl_a)
    
    assert summary['geometry_type'] == 'CYLINDER'
    assert summary['datum_label'] == 'B'
    assert summary['parent'] == 'PlaneA'
    
    print(f"  Tolerance summary for {cyl_a}:")
    for key, value in summary.items():
        print(f"    {key}: {value}")
    print("✓ Tolerance summary successful\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING REFACTORED COMPONENT CLASS")
    print("=" * 60 + "\n")
    
    try:
        test_component_initialization()
        test_geometry_relationships()
        test_datum_management()
        test_convenience_methods()
        test_dimension_tracking()
        test_tolerance_management()
        test_geometry_intersection()
        test_attribute_access()
        test_tolerance_summary()
        
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