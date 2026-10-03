"""Standardized data record and provenance schema for Phase 1 data collection."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class DocumentRecord:
    """Standardized document record with strict provenance tracking."""
    doc_id: str
    language: str  # "hi" or "ne"
    source_type: str  # "manual" or "downloaded"
    source_name: str  # e.g., "indic_hplt_v2", "hindi_wikipedia", "dainik_jagran"
    text: str
    url: Optional[str] = None
    title: Optional[str] = None
    collection_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    raw_location: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentRecord":
        """Create record from dictionary."""
        return cls(
            doc_id=str(data.get("doc_id", "")),
            language=str(data.get("language", "")),
            source_type=str(data.get("source_type", "")),
            source_name=str(data.get("source_name", "")),
            text=str(data.get("text", "")),
            url=data.get("url"),
            title=data.get("title"),
            collection_date=data.get(
                "collection_date", datetime.now(timezone.utc).isoformat()
            ),
            raw_location=data.get("raw_location"),
            metadata=data.get("metadata", {}),
        )


def validate_record(record: Dict[str, Any]) -> bool:
    """Validate that a record conforms to the required provenance schema."""
    required_keys = ["doc_id", "language", "source_type", "source_name", "text", "collection_date"]
    for key in required_keys:
        if key not in record or record[key] is None or record[key] == "":
            return False

    if record["language"] not in ("hi", "ne"):
        return False

    if record["source_type"] not in ("manual", "downloaded"):
        return False

    return True
