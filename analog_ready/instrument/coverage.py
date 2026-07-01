"""What the instrumentation actually touched. The denominator (`modules_total`) is the count of
ELIGIBLE targets, not every module — so the report can never overstate coverage. Linear-like
modules we cannot safely wrap (e.g. a fused-QKV Conv1D) are disclosed, not silently dropped."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CoverageReport:
    model: str
    modules_total: int = 0          # eligible targets (nn.Linear + nn.Conv2d) seen
    linear_replaced: int = 0
    conv_replaced: int = 0
    attention_projection_match_rate: float = 1.0
    fused_qkv_policy: str = "none"
    unhandled_linear_like: list = field(default_factory=list)
    coverage_confidence: str = "medium"

    @property
    def linear_modules_replaced(self) -> int:
        """Alias for linear_replaced (public surface name used by W4 tests)."""
        return self.linear_replaced

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "modules_total": self.modules_total,
            "linear_replaced": self.linear_replaced,
            "conv_replaced": self.conv_replaced,
            "attention_projection_match_rate": self.attention_projection_match_rate,
            "fused_qkv_policy": self.fused_qkv_policy,
            "unhandled_linear_like": list(self.unhandled_linear_like),
            "coverage_confidence": self.coverage_confidence,
        }
