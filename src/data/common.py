"""Shared paths and small helpers."""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..config import RAW_TARGETS as TARGETS  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
GLORIA_DIR = RAW / "gloria" / "GLORIA_2022"
SRF_DIR = RAW / "srf"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs() -> None:
    for d in (SRF_DIR, INTERIM, PROCESSED):
        d.mkdir(parents=True, exist_ok=True)
