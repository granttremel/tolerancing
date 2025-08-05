# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python library for handling complex tolerancing calculations in optical systems. The library provides a framework for:
- Creating components with datums (geometric features like points, axes, planes, cylinders, spheres)
- Assembling components through mating relationships
- Building datum transfer trees that connect datums across different components
- Calculating tolerance stackups with dimension-aware contributions
- Generating tolerance analysis reports

## Architecture

### Core Classes

- **Geometry** (tolerancing/geometry.py): Enum defining geometric types (POINT, AXIS, PLANE, CYLINDER, SPHERE, etc.) with dimension calculations
- **Datum** (tolerancing/datum.py): Represents geometric features with position, orientation, and tolerance properties
- **Component** (tolerancing/component.py): Contains multiple datums and manages their relationships
- **Assembly** (tolerancing/assembly.py): Manages components and their mating relationships, generates datum transfer maps
- **GeoCalculator** (tolerancing/geometry.py): Provides geometric calculations (distance, intersection, parallelism) between datums

### Key Concepts

1. **Datum Transfer**: The process of relating datums between mated components to track how tolerances propagate through an assembly
2. **Tolerance Stackup**: Accumulation of tolerances through a series of datum transfers, accounting for both min/max and statistical contributions
3. **Dimension-Aware Tolerances**: Tolerances that scale based on geometric relationships (e.g., flatness error proportional to distance)

## Development Commands

```bash
# Install the package in development mode
pip install -e .

# Run tests (when available)
python scripts/test.py

# The project uses a virtual environment at tol-env/
source tol-env/bin/activate  # Linux/Mac
# or
tol-env\Scripts\activate  # Windows
```

## Implementation Status

The project is in early development. Current implementation includes:
- Basic class structure for Assembly, Component, Datum, and Geometry
- Partial implementation of GeoCalculator for distance and intersection calculations
- Framework for datum relationships and mating

## Key Implementation Notes

- All spatial coordinates use numpy arrays
- Datums have both relative (to parent component) and absolute coordinates
- The geometry type determines required parameters (e.g., circular geometries need radius)
- Component datums are accessed via dictionary or attribute notation (e.g., `component.datums['A']` or `component.A`)