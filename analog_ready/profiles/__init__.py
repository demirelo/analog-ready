"""Named hardware profiles. Each YAML carries the cost coefficients the break-even model needs,
with the per-field redaction from W1 preserved — so a vendor can swap in a private profile and the
shareable report still never leaks a `hidden` coefficient."""
from __future__ import annotations

from pathlib import Path

import yaml

from analog_ready.core.profile import HardwareProfile, ProfileField

_DIR = Path(__file__).parent


def _profile_from_data(data: dict, default_name: str) -> HardwareProfile:
    """Build a HardwareProfile from parsed YAML, preserving the FULL W11 professional schema
    (unit/source/uncertainty/provenance/calibration_date) — not just value/redaction. The single
    place profile YAML becomes a profile, so a builtin name and a custom path can never diverge."""
    fields = {
        key: ProfileField(
            value=spec["value"], redaction=spec.get("redaction", "hidden"),
            unit=spec.get("unit", ""), source=spec.get("source", ""),
            uncertainty=spec.get("uncertainty", ""),
            provenance=spec.get("provenance", "literature"),
            calibration_date=spec.get("calibration_date", ""))
        for key, spec in (data.get("fields") or {}).items()
    }
    return HardwareProfile(name=data.get("name", default_name),
                           visibility=data.get("visibility", "private"), fields=fields)


def load_profile_from_path(path) -> HardwareProfile:
    """Load a profile from an explicit YAML file path (a vendor's custom private profile).
    Preserves the same professional schema as the builtin loader."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    return _profile_from_data(data, path.stem)


def load_profile(name: str) -> HardwareProfile:
    path = _DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"unknown profile {name!r} (looked for {path})")
    return _profile_from_data(yaml.safe_load(path.read_text()) or {}, name)
