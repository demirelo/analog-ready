"""HardwareProfile with per-field redaction. A profile is a vendor's crown-jewel IP, so every
field declares how it may leave the tool: exact_ok (share as-is), bucket (coarsen), or hidden
(never expose). `redacted()` is what a local-only run is allowed to emit."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

REDACTIONS = ("exact_ok", "bucket", "hidden")
PROVENANCES = ("literature", "measured", "estimate")


@dataclass
class ProfileField:
    value: Any
    redaction: str = "hidden"
    # W11 professional schema — optional provenance metadata a hardware team needs to trust a
    # coefficient. All optional with defaults, so a bare {value[, redaction]} field still loads.
    unit: str = ""                     # e.g. "pJ", "pJ/byte", "bits"
    source: str = ""                   # citation / datasheet / measurement report
    uncertainty: str = ""              # e.g. "+-50%", "1-sigma 0.3"
    provenance: str = "literature"     # one of PROVENANCES
    calibration_date: str = ""         # ISO date the value was measured/calibrated, if any

    def __post_init__(self) -> None:
        if self.redaction not in REDACTIONS:
            raise ValueError(
                f"unknown redaction {self.redaction!r}; expected one of {REDACTIONS}")
        if self.provenance not in PROVENANCES:
            raise ValueError(
                f"unknown provenance {self.provenance!r}; expected one of {PROVENANCES}")

    def __repr__(self) -> str:
        # Defensive: a `hidden` field's raw value must not leak via repr/str (logs, tracebacks).
        if self.redaction == "exact_ok":
            shown = repr(self.value)
        elif self.redaction == "bucket":
            shown = repr(_bucket(self.value))
        else:  # hidden
            shown = "<hidden>"
        return f"ProfileField(value={shown}, redaction={self.redaction!r})"


def _bucket(value: Any) -> str:
    """Coarsen a value to a non-exact, non-empty summary. Numbers collapse to an order-of-magnitude
    band so the exact figure never leaves the tool."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        if not math.isfinite(value):  # NaN/inf must not abort the whole redacted report
            return "non-finite"
        if value == 0:
            return "~0"  # coarsened: never emit the exact figure, even for zero
        lo = 10 ** math.floor(math.log10(abs(value)))
        if value < 0:  # band must bracket the value and stay monotonic, e.g. -5 -> "-10..-1"
            return f"-{lo * 10:g}..-{lo:g}"
        return f"{lo:g}..{lo * 10:g}"
    return "bucketed"


@dataclass
class HardwareProfile:
    name: str = "default"
    visibility: str = "private"
    fields: dict[str, ProfileField] = field(default_factory=dict)

    def redacted(self) -> dict:
        """The shareable view. exact_ok -> raw value; bucket -> coarsened band; hidden -> omitted
        entirely (its raw value must never appear anywhere in the output)."""
        out: dict[str, Any] = {}
        for key, fld in self.fields.items():
            if fld.redaction == "exact_ok":
                out[key] = fld.value
            elif fld.redaction == "bucket":
                out[key] = _bucket(fld.value)
            # "hidden": intentionally omitted
        return out
