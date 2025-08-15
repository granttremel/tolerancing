#!/usr/bin/env python3
"""Debug datum label issue."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tolerancing import Component

# Create component
comp = Component("TestPart")

# Add geometries with explicit datum labels
plane_b_id = comp.add_plane(reference_id="PlaneA", offset=10, datum_label="B")
axis_a_id = comp.add_axis(reference_id="PlaneA", datum_label="C")

print(f"PlaneB ID: {plane_b_id}")
print(f"Datum label for PlaneB: {comp.get_datum_label(plane_b_id)}")
print(f"All datums: {comp.datums}")
print(f"All geometries: {comp.geometries.keys()}")