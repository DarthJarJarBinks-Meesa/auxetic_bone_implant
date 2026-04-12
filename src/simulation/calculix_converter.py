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


def _parse_abaqus_element_type_from_header(element_header_line: str) -> str | None:
    """Return the ``TYPE=`` value from an ``*ELEMENT`` header line, or None."""
    for part in element_header_line.split(","):
        kv = part.strip().split("=", 1)
        if len(kv) == 2 and kv[0].strip().upper() == "TYPE":
            return kv[1].strip()
    return None


def _is_abaqus_c3d_volume_solid(etype: str) -> bool:
    """True for 3D continuum solids (C3D4, C3D8, C3D10, …) written for solid mechanics."""
    return etype.upper().startswith("C3D")


def strip_non_c3d_element_blocks_from_abaqus_inp(inp_path: Path) -> int:
    """
    Remove ``*ELEMENT`` sections whose type is not ``C3D*``.

    Gmsh's Abaqus export often adds T3D2 (lines) and CPS3 (plane-stress faces).
    Those need different CalculiX sections; our deck only assigns ``*SOLID SECTION``
    to ``ALL_ELEMS`` (volume tets/bricks). Stripping non-C3D blocks avoids
    ``gen3delem`` / zero-thickness errors.

    Returns:
        Number of ``*ELEMENT`` sections removed.
    """
    raw = inp_path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    removed = 0
    while i < len(lines):
        line = lines[i]
        stripped_up = line.strip().upper()
        if stripped_up.startswith("*ELEMENT"):
            etype = _parse_abaqus_element_type_from_header(line)
            if etype is not None and not _is_abaqus_c3d_volume_solid(etype):
                removed += 1
                i += 1
                while i < len(lines) and not lines[i].lstrip().startswith("*"):
                    i += 1
                continue
        out.append(line)
        i += 1
    inp_path.write_text("".join(out), encoding="utf-8")
    if removed:
        logger.info(
            "Stripped %d non-C3D *ELEMENT section(s) from %s for CalculiX compatibility.",
            removed,
            inp_path.name,
        )
    return removed

class ConversionError(Exception):
    pass


def format_set(set_type: str, name: str, tags: list[int]) -> str:
    """Format list of IDs into Abaqus/Calculix NSET/ELSET blocks (max 16 items per line)."""
    if not tags:
        return ""
    
    lines = [f"*{set_type}, {set_type}={name}"]
    for i in range(0, len(tags), 16):
        lines.append(", ".join(str(t) for t in tags[i:i+16]))
    return "\n".join(lines) + "\n"


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
        # gmsh.model.mesh.clear expects a list of (dim, tag) entity pairs, not bare
        # integers; stripping via API is also brittle for mesh-only models. We
        # instead post-process the written Abaqus file to drop non-C3D *ELEMENT
        # sections (T3D2, CPS3, …).
        gmsh.write(str(output_inp))
        strip_non_c3d_element_blocks_from_abaqus_inp(output_inp)
        
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
            f.write("\n".join(additions))
    except Exception as e:
        raise ConversionError(f"Failed to append sets to {output_inp}: {e}")
        
    logger.info(
        "Mesh converted to CalculiX .inp format (Nodes: %d, Elements: %d, Path: %s)",
        len(all_nodes), len(all_elem_tags), str(output_inp)
    )
    
    return str(output_inp)
