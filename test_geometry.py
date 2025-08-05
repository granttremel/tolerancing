#!/usr/bin/env python3
"""Test script for geometry operations"""

import numpy as np
from tolerancing.geometry.geometry import _Geometry, GeometryType, GeoCalculator
# from tolerancing.datum import Geometry

# Create calculator instance
calc = GeoCalculator()

print("Testing geometry operations...")
print("="*50)

# Test 1: Point-Point distance
print("\n1. Point-Point Distance:")
p1 = _Geometry(GeometryType.POINT, [0, 0, 0])
p2 = _Geometry(GeometryType.POINT, [3, 4, 0])
dist = calc.distance(p1, p2)
print(f"   Distance between {p1.aorigin} and {p2.aorigin}: {dist}")
print(f"   Expected: 5.0, Got: {dist}")

# Test 2: Point-Axis distance
print("\n2. Point-Axis Distance:")
p = _Geometry(GeometryType.POINT, [2, 2, 0])
ax = _Geometry(GeometryType.AXIS, [0, 0, 0], [1, 0, 0])  # X-axis
dist = calc.distance(p, ax)
print(f"   Distance from point {p.aorigin} to X-axis: {dist}")
print(f"   Expected: 2.0, Got: {dist}")

# Test 3: Axis-Axis intersection
print("\n3. Axis-Axis Intersection:")
ax1 = _Geometry(GeometryType.AXIS, [0, 0, 0], [1, 0, 0])  # X-axis
ax2 = _Geometry(GeometryType.AXIS, [0, 0, 0], [0, 1, 0])  # Y-axis
intersection = calc.intersect(ax1, ax2)
if intersection:
    print(f"   Axes intersect at: {intersection.aorigin}")
    print(f"   Expected: [0,0,0], Got: {intersection.aorigin}")

# Test 4: Axis-Plane intersection
print("\n4. Axis-Plane Intersection:")
ax = _Geometry(GeometryType.AXIS, [1, 1, 1], [1, 1, 1])  # Diagonal axis
pl = _Geometry(GeometryType.PLANE, [0, 0, 3], [0, 0, 1])  # Z=3 plane
intersection = calc.intersect(ax, pl)
if intersection:
    print(intersection)
    print(f"   Axis intersects plane at: {intersection.aorigin}")
    print(f"   Expected: [3,3,3], Got: {intersection.aorigin}")

# Test 5: Plane-Plane intersection
print("\n5. Plane-Plane Intersection:")
pl1 = _Geometry(GeometryType.PLANE, [0, 0, 0], [1, 0, 0])  # YZ plane
pl2 = _Geometry(GeometryType.PLANE, [0, 0, 0], [0, 1, 0])  # XZ plane
intersection = calc.intersect(pl1, pl2)
if intersection:
    print(f"   Planes intersect in axis with:")
    print(f"   Origin: {intersection.aorigin}")
    print(f"   Direction: {intersection.aframe}")
    print(f"   Expected direction: [0,0,1] (Z-axis)")

# Test 6: Point-Sphere distance
print("\n6. Point-Sphere Distance:")
p = _Geometry(GeometryType.POINT, [5, 0, 0])
sph = _Geometry(GeometryType.SPHERE, [0, 0, 0], r=3)
dist_min = calc.distance(p, sph, mode="min")
dist_center = calc.distance(p, sph, mode="center")
print(f"   Min distance to sphere: {dist_min}")
print(f"   Center distance to sphere: {dist_center}")
print(f"   Expected min: 2.0, center: 5.0")

# Test 7: Sphere-Sphere intersection
print("\n7. Sphere-Sphere Intersection:")
sph1 = _Geometry(GeometryType.SPHERE, [0, 0, 0], r=5)
sph2 = _Geometry(GeometryType.SPHERE, [8, 0, 0], r=5)
intersection = calc.intersect(sph1, sph2)
if intersection:
    print(f"   Spheres intersect in circle with:")
    print(f"   Center: {intersection.aorigin}")
    print(f"   Normal: {intersection.aframe}")
    print(f"   Radius: {intersection.r}")

# Test 8: Contains check
print("\n8. Contains Check:")
p = _Geometry(GeometryType.POINT, [1, 2, 0])
cyl = _Geometry(GeometryType.CYLINDER, [0, 0, 0], [0, 0, 1], r=3)  # Z-axis cylinder
contains = calc.contains(p, cyl)
print(f"   Cylinder contains point: {contains}")
print(f"   Expected: True (distance {np.sqrt(5)} < radius 3)")

print("\n" + "="*50)
print("Tests completed!")