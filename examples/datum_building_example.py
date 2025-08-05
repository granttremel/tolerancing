#!/usr/bin/env python3
"""
Example demonstrating the parametric datum building system.

This shows how a mechanical engineer would build up datums step by step,
starting with primary datums and creating derived datums through dimensional relationships.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tolerancing.datum import Datum, Tolerance, ToleranceType
from tolerancing.geometry import GeometryType

def main():
    print("=== Parametric Datum Building Example ===\n")
    
    # Step 1: Create primary datums (these are "fixed" features)
    print("1. Creating primary datums:")
    
    # Primary plane datum (like the bottom face of a part)
    datum_A = Datum.from_geometry(
        geometry_type=GeometryType.PLANE,
        origin=[0, 0, 0], 
        frame=[0, 0, 1],  # Normal pointing up
        name="A"
    )
    print(f"   Primary datum A: {datum_A}")
    
    # Primary axis datum (like a cylindrical bore)
    datum_B = Datum.from_geometry(
        geometry_type=GeometryType.AXIS,
        origin=[0, 0, 0],
        frame=[0, 0, 1],  # Axis pointing up
        name="B"  
    )
    print(f"   Primary datum B: {datum_B}")
    
    # Step 2: Create derived datums using dimensions
    print("\n2. Creating derived datums from dimensions:")
    
    # Offset plane 25mm above datum A
    datum_A1 = datum_A.offset_distance(
        distance=25.0,
        direction=[0, 0, 1],  # Up direction
        target_geometry=GeometryType.PLANE,
        name="A1"
    )
    print(f"   Derived plane A1 (25mm above A): {datum_A1}")
    
    # Point offset in X,Y from intersection of A and B
    point_C = datum_A.offset_xy(
        dx=50.0, 
        dy=30.0,
        target_geometry=GeometryType.POINT,
        name="C"
    )
    print(f"   Derived point C (50,30 from A): {point_C}")
    
    # Axis at 45° angle from datum B
    axis_D = datum_B.angle_from(
        angle=np.pi/4,  # 45 degrees
        reference_direction=[1, 0, 0],  # X direction as reference
        target_geometry=GeometryType.AXIS,
        name="D"
    )
    print(f"   Derived axis D (45° from B): {axis_D}")
    
    # Step 3: Create more complex derived datums
    print("\n3. Creating complex derived datums:")
    
    # Create a second plane for intersection
    datum_E = Datum.from_geometry(
        geometry_type=GeometryType.PLANE,
        origin=[10, 0, 0],
        frame=[1, 0, 0],  # Normal in X direction
        name="E"
    )
    
    # Axis from intersection of two planes
    intersection_axis = Datum.intersect_planes(datum_A, datum_E, name="intersection")
    print(f"   Intersection axis of A and E: {intersection_axis}")
    
    # Step 4: Show construction dependencies
    print("\n4. Construction dependencies:")
    
    print(f"   A1 depends on: {[d.name for d in datum_A1.get_construction_chain()]}")
    print(f"   Point C depends on: {[d.name for d in point_C.get_construction_chain()]}")
    print(f"   Intersection depends on: {[d.name for d in intersection_axis.get_construction_chain()]}")
    
    # Step 5: Add tolerances to dimensions
    print("\n5. Adding tolerances:")
    
    # Add dimensional tolerance to the 25mm offset
    if datum_A1.dimension:
        datum_A1.dimension.set_tolerance(Tolerance(ToleranceType.DIMENSION, 0.1, 0.1))
        print(f"   Added ±0.1mm tolerance to 25mm dimension")
    
    # Add position tolerance to point C
    point_C.set_tolerance(Tolerance(ToleranceType.POSITION, 0.05))
    print(f"   Added ⌖0.05 position tolerance to point C")
    
    # Step 6: Show validation
    print("\n6. Datum validation:")
    
    print(f"   Datum A defined: {datum_A.is_defined()}")
    print(f"   Datum A1 defined: {datum_A1.is_defined()}")
    print(f"   Point C defined: {point_C.is_defined()}")
    
    print("\n=== Example Complete ===")
    print("This demonstrates how datums can be built parametrically,")
    print("just like a mechanical engineer would dimension a part!")

if __name__ == "__main__":
    main()