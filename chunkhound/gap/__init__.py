"""Git-agnostic semantic gap analysis engine (stateless).

`chunkhound gap` compares two directory trees (A and B) and emits a deterministic
`gap.v1` JSON changeset describing file- and symbol-level changes.
"""

from chunkhound.gap.engine import GapEngine

__all__ = ["GapEngine"]

