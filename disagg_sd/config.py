from dataclasses import dataclass, fields
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    policy: str = "minedraft"
    draft_device: int = 1
    max_ahead: int = 64
    draft_timeout: float = 60.0
    trace_path: str | None = None

    @classmethod
    def parse(cls, values: dict[str, Any]) -> "RuntimeConfig":
        unknown = values.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown disagg_sd options: {sorted(unknown)}")
        config = cls(**values)
        if config.policy not in ("minedraft", "overlap", "sync"):
            raise ValueError("policy must be 'minedraft', 'overlap', or 'sync'")
        if type(config.draft_device) is not int or config.draft_device < 0:
            raise ValueError("draft_device must be a nonnegative device index")
        if type(config.max_ahead) is not int or config.max_ahead < 1:
            raise ValueError("max_ahead must be a positive integer")
        if (
            type(config.draft_timeout) not in (int, float)
            or not isfinite(config.draft_timeout)
            or config.draft_timeout <= 0
        ):
            raise ValueError("draft_timeout must be finite and positive")
        if config.trace_path is not None and not isinstance(config.trace_path, str):
            raise ValueError("trace_path must be a path string")
        return config
