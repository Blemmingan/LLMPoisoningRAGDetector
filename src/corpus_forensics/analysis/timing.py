"""Wall-clock time per pipeline stage (computational cost analysis)."""

import time
from collections.abc import Iterator
from contextlib import contextmanager


class StageTimer:
    """Accumulates seconds per stage name; reset per document."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    def reset(self) -> None:
        self.seconds = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start

    def snapshot(self) -> dict[str, float]:
        return {k: round(v, 5) for k, v in self.seconds.items()}
