"""The break-even envelope — the buyer-facing 'where does analog stop winning?'. Three constraints,
each a profile-parameter threshold the op must clear: converter energy per MAC must undercut the
digital MAC it replaces (X), weight reuse must amortise analog programming (Y), and the required
ENOB must fit the analog dynamic range (Z = enob_avail). `sweep` varies one profile parameter and
shows the verdict flip — without that flip the report is just a noise benchmark."""
from __future__ import annotations

import math
from dataclasses import dataclass

from analog_ready.cost_model import estimate_op

def reuse_threshold(profile) -> float:
    """Y: the MACs-per-weight needed to amortise the analog weight-programming (write) energy. Derived
    from the profile's programming-energy coefficient, not a magic constant — favorable analog needs
    weight reuse above this."""
    program = float(profile.fields["weight_program_energy_pj"].value)
    digital_mac = float(profile.fields["digital_mac_energy_pj"].value)
    return program / digital_mac if digital_mac else 0.0


@dataclass
class BreakEven:
    name: str
    favorable: bool
    converter_ok: bool
    reuse_ok: bool
    precision_ok: bool
    x_converter_pj_per_mac: float   # threshold: analog wins only below this converter pJ/MAC
    y_reuse_threshold: float
    z_enob_avail: float
    converter_pj_per_mac: float     # this op's actual
    reuse: float
    enob_req: float

    def to_dict(self) -> dict:
        return {
            "favorable": self.favorable,
            "converter_ok": self.converter_ok, "reuse_ok": self.reuse_ok,
            "precision_ok": self.precision_ok,
            "thresholds": {"x_converter_pj_per_mac": self.x_converter_pj_per_mac,
                           "y_reuse": self.y_reuse_threshold, "z_enob_avail": self.z_enob_avail},
            "actual": {"converter_pj_per_mac": self.converter_pj_per_mac,
                       "reuse": self.reuse, "enob_req": self.enob_req},
        }


@dataclass
class SweepPoint:
    value: float
    favorable: bool
    converter_ok: bool
    reuse_ok: bool
    precision_ok: bool

    def to_dict(self) -> dict:
        return {"value": self.value, "favorable": self.favorable,
                "converter_ok": self.converter_ok, "reuse_ok": self.reuse_ok,
                "precision_ok": self.precision_ok}


def enob_required(feat, profile) -> float:
    """Conservative precision envelope: B_in + 0.5*log2(K). The formula follows arXiv:2405.14978
    (its Eq. 2, ADCres = B_i + log2(k*FS*sqrt(D_i)), reduces to B_i + 0.5*log2(D_i) at k=2, FS=0.5);
    arXiv:2401.15061 supports the qualitative ADC-resolution pressure in analog MVMs, not this
    exact form."""
    return float(profile.fields["input_bits"].value) + 0.5 * math.log2(max(feat.K, 1))


def break_even(feat, profile) -> BreakEven:
    cost = estimate_op(feat, profile)
    x = float(profile.fields["digital_mac_energy_pj"].value)
    converter_pj_per_mac = cost.converter_pj_per_mac
    # a zero-MAC op converts/computes nothing — it can claim no converter advantage
    converter_ok = feat.macs > 0 and converter_pj_per_mac < x

    reuse = feat.weight_reuse
    y = reuse_threshold(profile)
    reuse_ok = reuse > y

    enob_req = enob_required(feat, profile)
    z = float(profile.fields["enob_avail"].value)
    precision_ok = enob_req < z

    favorable = converter_ok and reuse_ok and precision_ok
    return BreakEven(feat.name, favorable, converter_ok, reuse_ok, precision_ok,
                     x, y, z, converter_pj_per_mac, reuse, enob_req)


def _with_field(profile, key: str, value):
    from analog_ready.core.profile import HardwareProfile, ProfileField

    new_fields = dict(profile.fields)
    old = new_fields.get(key)
    if old is not None:
        # Same field, new value: carry the full professional schema (unit/source/uncertainty/
        # provenance/calibration_date) so a swept profile keeps its metadata (matches the loader).
        new_fields[key] = ProfileField(value=value, redaction=old.redaction, unit=old.unit,
                                       source=old.source, uncertainty=old.uncertainty,
                                       provenance=old.provenance,
                                       calibration_date=old.calibration_date)
    else:
        new_fields[key] = ProfileField(value=value, redaction="exact_ok")
    return HardwareProfile(name=profile.name, visibility=profile.visibility, fields=new_fields)


def sweep(feat, profile, param: str, values) -> list:
    pts = []
    for v in values:
        be = break_even(feat, _with_field(profile, param, v))
        pts.append(SweepPoint(v, be.favorable, be.converter_ok, be.reuse_ok, be.precision_ok))
    return pts
