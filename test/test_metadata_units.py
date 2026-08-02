"""Consistency checks for the FA_METADATA units table.

Several entries in FA_METADATA state the physical unit twice: once inside the
``long_name`` text and once in the ``units`` field.  When an entry is added by
copying a neighbouring one it is easy to update the ``long_name`` and forget the
``units`` -- which is how VITESSE_VERT (omega, Pa/s) ended up advertising
'm s-1', the unit of its sibling VERT_VELOCIT.

These tests pin the units of the fields whose long_name declares one, so the
same copy-paste slip cannot come back unnoticed.
"""

import re

import pytest

from faxarray.fa_metadata import FA_METADATA, get_metadata


# Units that the long_name may spell differently from the CF-style units field.
_ALIASES = {
    "m/s": "m s-1",
    "pa/s": "pa s-1",
    "mm/h": "mm h-1",
    "k/s": "k s-1",
    "dbz": "dbz",
}


def _normalise(unit: str) -> str:
    key = unit.strip().lower()
    key = _ALIASES.get(key, key)
    return key.replace(" ", "").replace("^", "").replace("**", "")


# Fields whose long_name embeds a real unit, and the units they must declare.
EXPECTED_UNITS = {
    "VERT.VELOCIT": "m s-1",
    "VERT_VELOCIT": "m s-1",
    "VITESSE_VERT": "Pa s-1",
    "REFLEC_DBZ": "dBZ",
    "REFLECT_DBZ.MAX": "dBZ",
    "REFLECT_DBZ_MAX": "dBZ",
    "SIM_REFLECTI": "mm h-1",
    "SURFREFLECT.MAX": "mm h-1",
    "SURFREFLECT_MAX": "mm h-1",
}


@pytest.mark.parametrize("field,expected", sorted(EXPECTED_UNITS.items()))
def test_declared_units_match_long_name(field, expected):
    assert field in FA_METADATA, f"{field} missing from FA_METADATA"
    assert FA_METADATA[field]["units"] == expected


def test_vitesse_vert_is_a_pressure_velocity():
    """omega = DPi/Dt is in Pa/s -- not the m/s of its VERT_VELOCIT sibling."""
    meta = get_metadata("VITESSE_VERT")
    assert meta["units"] == "Pa s-1"
    assert "omega" in meta["long_name"]


def test_vert_velocit_and_vitesse_vert_do_not_share_units():
    """The two vertical velocity fields are the same motion in different units."""
    assert get_metadata("VERT_VELOCIT")["units"] != get_metadata("VITESSE_VERT")["units"]


def test_level_prefixed_names_resolve_to_the_same_units():
    """S051VITESSE_VERT and friends must inherit the corrected units."""
    for name in ("S051VITESSE_VERT", "S001VITESSE_VERT", "S087VITESSE_VERT"):
        assert get_metadata(name)["units"] == "Pa s-1", name
    for name in ("S051VERT.VELOCIT", "S051VERT_VELOCIT"):
        assert get_metadata(name)["units"] == "m s-1", name


# Parenthesised qualifiers that are not units, e.g. 'Maximum of CLWC (60min)'.
_NOT_A_UNIT = re.compile(
    r"^(60min|small|large|AROME|CANA|Water|A|B|C|\*g)$", re.IGNORECASE
)
_TRAILING_PAREN = re.compile(r"\(([^)]*)\)\s*$")


def _looks_like_a_unit(text: str) -> bool:
    """True for 'm/s', 'Pa/s', 'dBZ', 'kg kg-1'; false for 'relative humidity'.

    Unit symbols are short tokens; prose in parentheses is not.
    """
    tokens = text.split()
    return bool(tokens) and all(len(tok) <= 4 for tok in tokens)


def test_no_entry_contradicts_its_own_long_name():
    """Sweep the whole table for units that disagree with the long_name text."""
    offenders = []
    for field, meta in FA_METADATA.items():
        match = _TRAILING_PAREN.search(meta.get("long_name", "").strip())
        if not match:
            continue
        claimed = match.group(1).strip()
        if _NOT_A_UNIT.match(claimed) or not re.search(r"[a-zA-Z]", claimed):
            continue
        if not _looks_like_a_unit(claimed):
            continue
        if _normalise(claimed) != _normalise(meta.get("units", "")):
            offenders.append((field, claimed, meta.get("units")))
    assert not offenders, "units contradict long_name for: " + ", ".join(
        f"{f} (long_name says {c!r}, units={u!r})" for f, c, u in offenders
    )
