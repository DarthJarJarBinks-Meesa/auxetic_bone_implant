"""Tests for CalculiX .dat table parsing (*NODE PRINT / *EL PRINT)."""

import math

from analysis.postprocess import (
    PostprocessArtifacts,
    extract_scalar_results_from_artifacts,
    parse_max_displacement_from_dat_print_tables,
    parse_max_von_mises_from_dat_print_tables,
    von_mises_from_cauchy_voigt,
)

def test_von_mises_pure_sxx() -> None:
    assert abs(von_mises_from_cauchy_voigt(200.0, 0.0, 0.0, 0.0, 0.0, 0.0) - 200.0) < 1e-9


def test_von_mises_shear() -> None:
    vm = von_mises_from_cauchy_voigt(0.0, 0.0, 0.0, 0.0, 0.0, 100.0)
    assert abs(vm - 100.0 * math.sqrt(3.0)) < 1e-6


def test_parse_displacement_table() -> None:
    dat = """
displacements (vx,vy,vz) for set ALL_NODES and time  1.0000000E+00

       1 -1.0000000E-03  0.0000000E+00  0.0000000E+00
       2  0.0000000E+00  3.0000000E-03  0.0000000E+00

stresses (xx,yy,zz,xy,xz,yz) for set ALL_ELEMS and time  1.0000000E+00

       1  1  3.0000000E+02 -1.0000000E+02  5.0000000E+01  0.0000000E+00  0.0000000E+00  0.0000000E+00
"""
    assert abs(parse_max_displacement_from_dat_print_tables(dat) - 3.0e-3) < 1e-12


def test_parse_stress_table_von_mises() -> None:
    dat = """
displacements (vx,vy,vz) for set ALL_NODES and time  1.

       1 0 0 0

stresses (xx,yy,zz,xy,xz,yz) for set ALL_ELEMS and time  1.

       1  1  3.0000000E+02 -1.0000000E+02  5.0000000E+01  0.0000000E+00  0.0000000E+00  0.0000000E+00
"""
    assert abs(parse_max_von_mises_from_dat_print_tables(dat) - 350.0) < 0.02


def test_extract_scalar_results_end_to_end(tmp_path) -> None:
    body = """
displacements (vx,vy,vz) for set ALL_NODES and time  1.
      10  0  0  0.004

stresses (xx,yy,zz,xy,xz,yz) for set ALL_ELEMS and time  1.
       1  1  200 0 0 0 0 0
"""
    p = tmp_path / "job.dat"
    p.write_text(body, encoding="utf-8")
    art = PostprocessArtifacts(dat_path=str(p), frd_path=str(tmp_path / "job.frd"))
    (tmp_path / "job.frd").write_bytes(b"\x00binary")
    r = extract_scalar_results_from_artifacts(art)
    assert r.metrics.get("max_displacement_mm") == 0.004
    assert abs(r.metrics.get("max_von_mises_stress_mpa") - 200.0) < 1e-9
    assert r.metrics.get("hotspot_stress_mpa") == r.metrics.get("max_von_mises_stress_mpa")
    assert not any("FRD" in w for w in r.warnings)
    assert not any("stress-strain" in w.lower() for w in r.warnings)


def test_table_parsers_joint_sample() -> None:
    dat = """
displacements (vx,vy,vz) for set ALL_NODES and time  1.
       9  0  0  0.002

stresses (xx,yy,zz,xy,xz,yz) for set ALL_ELEMS and time  1.
       3  1  100 0 0 0 0 0
"""
    assert parse_max_displacement_from_dat_print_tables(dat) == 0.002
    assert abs(parse_max_von_mises_from_dat_print_tables(dat) - 100.0) < 1e-9
