"""Document-level verdict policy.

A document is flagged as malicious when at least one finding passes the policy:
confidence at or above the threshold, optionally a verified fragment (found verbatim in
the document), and a minimum severity. The same policy is applied by the pipeline and,
over saved predictions, by the evaluation's threshold sweep.
"""

from pydantic import BaseModel, Field

from corpus_forensics.config import Settings
from corpus_forensics.schemas import Hallazgo, Severidad

SEVERITY_ORDER = {
    Severidad.BAJA: 0,
    Severidad.MEDIA: 1,
    Severidad.ALTA: 2,
    Severidad.CRITICA: 3,
}


class VerdictPolicy(BaseModel):
    """Which findings make a document malicious."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    require_verified: bool = True
    min_severity: Severidad = Severidad.BAJA

    @classmethod
    def from_settings(cls, settings: Settings) -> "VerdictPolicy":
        return cls(
            threshold=settings.verdict_threshold,
            require_verified=settings.verdict_require_verified,
            min_severity=Severidad(settings.verdict_min_severity.lower()),
        )

    def counts(self, finding: Hallazgo) -> bool:
        """Whether a finding is eligible regardless of its confidence."""
        if self.require_verified and not finding.verificado:
            return False
        return SEVERITY_ORDER[finding.severidad] >= SEVERITY_ORDER[self.min_severity]

    def score(self, findings: list[Hallazgo]) -> float:
        """Document score: highest confidence among eligible findings (0 if none)."""
        return max((f.confianza for f in findings if self.counts(f)), default=0.0)

    def is_malicious(self, findings: list[Hallazgo]) -> bool:
        return any(self.counts(f) for f in findings) and self.score(findings) >= self.threshold

    def describe(self) -> str:
        parts = [f"confianza >= {self.threshold:g}"]
        if self.require_verified:
            parts.append("fragmento verificado")
        if self.min_severity != Severidad.BAJA:
            parts.append(f"severidad >= {self.min_severity.value}")
        return ", ".join(parts)
