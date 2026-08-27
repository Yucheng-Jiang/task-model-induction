"""Compatibility exports for the canonical unified Step 6 contract.

Step 6 previously returned a pair of independently patched objective and
procedure models.  The production stage now returns one ``UnifiedTaskModel``.
The old public type names remain as aliases so downstream imports do not fail,
but they validate and serialize the canonical unified shape.
"""

from .unified_task_model import (
    UnifiedTaskModel,
    UnifiedTaskModelMergedMeta,
    UnifiedTaskModelOutput,
    UnifiedTaskModelRootResult,
)


# Backward-compatible names.  These are assignments rather than subclasses so
# callers get precisely the same parser/serializer and cannot accidentally
# revive the obsolete two-file model shape.
AlignedTaskModel = UnifiedTaskModel
BidirectionalAlignmentRootResult = UnifiedTaskModelRootResult
BidirectionalAlignmentMergedMeta = UnifiedTaskModelMergedMeta
BidirectionalAlignmentOutput = UnifiedTaskModelOutput
