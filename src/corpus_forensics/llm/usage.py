"""Token and cost accounting for paid model calls.

Every paid (or cached) call is appended to a local JSONL ledger. Before each paid call the
tracker checks the accumulated cost against a configurable cap, so a run can never spend
more than the team budget we decide to allow.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from corpus_forensics.llm.base import BudgetExceededError

# Approximate list prices in USD per 1M tokens, used only when the proxy does not report
# the real cost of a call (LiteLLM normally returns it in the `x-litellm-response-cost`
# header). Keys are matched against the model name without the provider prefix.
APPROX_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the USD cost of a call from the approximate price table (0.0 if unknown)."""
    name = model.split("/")[-1]
    for key, (price_in, price_out) in APPROX_PRICES_PER_MTOK.items():
        if name.startswith(key):
            return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000
    return 0.0


class UsageRecord(BaseModel):
    """One model call as written to the ledger."""

    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    kind: str = Field(description="'chat' or 'embedding'")
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cost_source: str = Field(default="none", description="'proxy', 'estimate' or 'cache'")
    cached: bool = False
    tag: str | None = Field(default=None, description="Free label, e.g. ablation config name")


class UsageTracker:
    """Appends usage records to a JSONL ledger and enforces a spending cap."""

    def __init__(self, ledger_path: Path | str, budget_limit_usd: float) -> None:
        self.ledger_path = Path(ledger_path)
        self.budget_limit_usd = budget_limit_usd
        self._total_cost = 0.0
        if self.ledger_path.exists():
            for record in self.read_records():
                self._total_cost += record.cost_usd

    @property
    def total_cost_usd(self) -> float:
        """Accumulated cost recorded in the local ledger."""
        return self._total_cost

    def check_budget(self) -> None:
        """Raise BudgetExceededError if the local ledger already reached the cap."""
        if self._total_cost >= self.budget_limit_usd:
            raise BudgetExceededError(
                f"Spending cap reached: {self._total_cost:.4f} USD recorded locally, "
                f"limit is {self.budget_limit_usd:.2f} USD (CF_BUDGET_LIMIT_USD)."
            )

    def record(self, record: UsageRecord) -> None:
        """Append a record to the ledger and update the running total."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
        self._total_cost += record.cost_usd

    def read_records(self) -> list[UsageRecord]:
        """Read every record from the ledger."""
        if not self.ledger_path.exists():
            return []
        records: list[UsageRecord] = []
        with open(self.ledger_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(UsageRecord.model_validate(json.loads(line)))
        return records
