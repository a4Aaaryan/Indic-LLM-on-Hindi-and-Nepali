"""Provenance tracking and inventory utilities for collected corpora."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from common.config import ensure_dir, resolve_path
from common.schema import DocumentRecord, validate_record


class ProvenanceTracker:
    """Manages document recording, provenance validation, and inventory summaries."""

    def __init__(self, language: str, output_file: str | Path):
        self.language = language
        self.output_file = resolve_path(output_file)
        ensure_dir(self.output_file.parent)
        self.counts_by_source: Dict[str, int] = defaultdict(int)
        self.counts_by_type: Dict[str, int] = defaultdict(int)
        self.total_docs = 0

    def write_record(self, record: DocumentRecord, file_handle) -> bool:
        """Validate and write a single record to an open JSONL file."""
        rec_dict = record.to_dict()
        if not validate_record(rec_dict):
            return False

        file_handle.write(json.dumps(rec_dict, ensure_ascii=False) + "\n")
        self.counts_by_source[record.source_name] += 1
        self.counts_by_type[record.source_type] += 1
        self.total_docs += 1
        return True

    def get_summary(self) -> Dict[str, Any]:
        """Return summary of tracked documents."""
        manual_count = self.counts_by_type.get("manual", 0)
        downloaded_count = self.counts_by_type.get("downloaded", 0)
        total = self.total_docs
        manual_pct = (manual_count / total * 100.0) if total > 0 else 0.0

        return {
            "language": self.language,
            "total_documents": total,
            "manual_documents": manual_count,
            "downloaded_documents": downloaded_count,
            "manual_percentage": round(manual_pct, 2),
            "by_source": dict(self.counts_by_source),
            "by_type": dict(self.counts_by_type),
        }
