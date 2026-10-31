"""On-disk response cache for paid model calls.

Identical requests (same model, parameters, schema and prompt) are answered from disk,
so re-running an evaluation or ablation does not pay twice for the same call.
"""

import hashlib
import json
from pathlib import Path
from typing import Any


class ResponseCache:
    """Simple content-addressed JSON cache stored under a directory."""

    def __init__(self, cache_dir: Path | str, enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

    @staticmethod
    def make_key(**parts: Any) -> str:
        """Build a stable hash from the request parts."""
        payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> Any | None:
        """Return the cached value for `key`, or None."""
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, value: Any) -> None:
        """Store `value` under `key`."""
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
