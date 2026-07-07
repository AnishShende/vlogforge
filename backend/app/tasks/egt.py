"""Pass 1.5 — EGT Document Assembly & Validation.

Assembles the final EGTDocument from all perception sub-stage outputs,
validates referential integrity, and persists to the job data store.
"""

import logging
from typing import List, Dict

from app.models import EGTSegment, EGTDocument

logger = logging.getLogger("VlogForge.EGT")


def build_egt_document(
    segments: List[EGTSegment],
    context_summary: str,
    source_file_count: int,
    total_duration: float,
    perception_model_version: str = "rule-based-v0",
) -> EGTDocument:
    """Assemble and validate a complete EGTDocument from perception outputs.

    Args:
        segments: Fully populated EGTSegment list (after ingest, transcribe, analyze, score).
        context_summary: Synthesized context document from analyze stage.
        source_file_count: Number of uploaded source files.
        total_duration: Total raw footage duration across all source files.
        perception_model_version: Model/version identifier for provenance.

    Returns:
        Validated EGTDocument.

    Raises:
        ValueError: If integrity validation fails with critical errors.
    """
    doc = EGTDocument(
        segments=segments,
        total_duration_sec=total_duration,
        source_file_count=source_file_count,
        context_summary=context_summary,
        perception_model_version=perception_model_version,
    )

    # Validate integrity
    errors = doc.validate_integrity()

    if errors:
        for err in errors:
            logger.error(f"EGT integrity error: {err}")
        raise ValueError(
            f"EGT document failed integrity validation with {len(errors)} error(s): "
            + "; ".join(errors[:5])
        )

    # Log summary statistics
    type_dist = {}
    bad_take_count = 0
    for seg in segments:
        type_dist[seg.segment_type] = type_dist.get(seg.segment_type, 0) + 1
        if seg.is_bad_take:
            bad_take_count += 1

    logger.info(
        f"EGT document assembled: "
        f"{len(segments)} segments, "
        f"{source_file_count} source files, "
        f"{total_duration:.1f}s total duration, "
        f"{bad_take_count} bad takes"
    )
    logger.info(f"EGT segment type distribution: {type_dist}")

    return doc


def egt_to_serializable(doc: EGTDocument) -> Dict:
    """Convert EGTDocument to a JSON-serializable dict for storage and API response."""
    return doc.model_dump()


def egt_from_serializable(data: Dict) -> EGTDocument:
    """Reconstruct an EGTDocument from a previously serialized dict."""
    return EGTDocument.model_validate(data)
