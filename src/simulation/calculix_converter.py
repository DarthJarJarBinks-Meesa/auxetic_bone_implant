"""
src/simulation/calculix_converter.py
======================================
Mesh conversion script to natively bridge Gmsh output (.msh) with the Calculix solver (.inp).

PIPELINE POSITION:
    mesher.py (model.msh) -> [THIS MODULE] -> model_mesh.inp -> case_runner.py -> solver_exporter.py

Extracted Node Sets:
    - ALL_NODES
    - FIXED_FACE (x <= x_min)
    - AXIAL_LOAD_FACE (x >= x_max)
    - BENDING_LOAD_FACE (top center strip, z >= z_max, x approx x_mid)
    - LEFT_SUPPORT (z <= z_min, x <= x_min)
    - RIGHT_SUPPORT (z <= z_min, x >= x_max)
"""

import logging
import os
from pathlib import Path
from typing import Any

import gmsh

logger = logging.getLogger(__name__)

class ConversionError(Exception):
    pass


def format_set(set_type: str, name: str, tags: list[int]) -> str:
    """Format list of IDs into Abaqus/Calculix NSET/ELSET blocks (max 16 items per line)."""
    if not tags:
        return ""
    
    lines = [f"*{set_type}, {set_type}={name}"]
    for i in range(0, len(tags), 16):
        lines.append(", ".join(str(t) for t in tags[i:i+16]))
    return "\\n".join(lines) + "\\n"


def convert_msh_to_inp(msh_path: str | Path) -> str:
    """
    Load a .msh file, export an Abaqus .inp, parse the geometry to identify boundary sets,
    and append the computed *NSET and *ELSET blocks to the file.
    
    Returns the path to the written _mesh.inp file.
    """
    msh_path = Path(msh_path)
    output_inp = msh_path.parent / f"{msh_path.stem}_mesh.inp"
    
    if not msh_path.exists():
        raise ConversionError(f"Mesh file not found: {msh_path}")
        
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    
    try:
        # Load mesh and export basic nodes/elements to INP
        gmsh.open(str(msh_path))
        gmsh.write(str(output_inp))
        
        # Extract nodal coordinates
        node_tags, coord, _ = gmsh.model.mesh.getNodes()
        if len(node_tags) == 0:
            raise ConversionError("No nodes found in mesh.")
            
        node_coords = []
        for i, tag in enumerate(node_tags):
            x = coord[i*3]
            y = coord[i*3 + 1]
            z = coord[i*3 + 2]
            node_coords.append((tag, x, y, z))
            
        # Extract element tags for ALL_ELEMS (only 3D volume elements)
        _, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
        all_elem_tags = []
        for tag_array in elem_tags:
            all_elem_tags.extend(tag_array)
            
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass

    # Geometry bounds analysis
    xmin = min(c[1] for c in node_coords)
    xmax = max(c[1] for c in node_coords)
    zmin = min(c[3] for c in node_coords)
    zmax = max(c[3] for c in node_coords)
    
    xmid = (xmax + xmin) / 2.0
    xband = (xmax - xmin) * 0.05
    tol = 1e-3  # 1um tolerance

    # Assign sets
    all_nodes = [c[0] for c in node_coords]
    fixed_nodes = [c[0] for c in node_coords if c[1] <= xmin + tol]
    axial_load_nodes = [c[0] for c in node_coords if c[1] >= xmax - tol]
    left_sup_nodes = [c[0] for c in node_coords if c[1] <= xmin + tol and c[3] <= zmin + tol]
    right_sup_nodes = [c[0] for c in node_coords if c[1] >= xmax - tol and c[3] <= zmin + tol]
    bending_nodes = [c[0] for c in node_coords if c[3] >= zmax - tol and abs(c[1] - xmid) <= xband + tol]

    # Quick validation fallback for bending if the mesh is too coarse at the exact midband
    if not bending_nodes:
        xband = (xmax - xmin) * 0.15 # expand to 15% width
        bending_nodes = [c[0] for c in node_coords if c[3] >= zmax - tol and abs(c[1] - xmid) <= xband + tol]

    # Build the INP string additions
    additions = [
        "** --- AUTOMATICALLY GENERATED BOUNDARY SETS ---",
        format_set("NSET", "ALL_NODES", all_nodes),
        format_set("ELSET", "ALL_ELEMS", all_elem_tags),
        format_set("NSET", "FIXED_FACE", fixed_nodes),
        format_set("NSET", "AXIAL_LOAD_FACE", axial_load_nodes),
        format_set("NSET", "BENDING_LOAD_FACE", bending_nodes),
        format_set("NSET", "LEFT_SUPPORT", left_sup_nodes),
        format_set("NSET", "RIGHT_SUPPORT", right_sup_nodes),
    ]
    
    # Append to exported file
    try:
        with open(output_inp, "a", encoding="utf-8") as f:
            f.write("\\n".join(additions))
    except Exception as e:
        raise ConversionError(f"Failed to append sets to {output_inp}: {e}")
        
    logger.info(
        "Mesh converted to CalculiX .inp format (Nodes: %d, Elements: %d, Path: %s)",
        len(all_nodes), len(all_elem_tags), str(output_inp)
    )
    
    return str(output_inp)
