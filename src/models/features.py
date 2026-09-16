"""Sensor feature sets and the band choices used by the empirical baselines.

Band choices (nearest available band centre; OLCI Oa2-Oa11 and MSI B1-B6 per prereg Amendment 1):
  role      hyperspectral  MSI (S2A)   OLCI (S3A)
  b443      hyp_445        msi_B1      olci_Oa3
  b490      hyp_490        msi_B2      olci_Oa4
  b510      hyp_510        (none)      olci_Oa5
  g560      hyp_560        msi_B3      olci_Oa6
  r665      hyp_665        msi_B4      olci_Oa8
  re705     hyp_705        msi_B5      olci_Oa11 (709 nm)
Blue bands used by the OC-type maximum band ratio: OC4-type {443, 490, 510}/560 for hyperspectral
and OLCI; OC3-type {443, 490}/560 for MSI (no 510 nm band).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

LOG_FLOOR = 1e-5  # sr^-1; non-positive Rrs are floored before log10 (5.3 % of hyp rows have a value <= 0)

SENSOR_PREFIX = {"hyp": "hyp_", "msi": "msi_", "olci": "olci_", "msi_strict": "msi_", "olci_strict": "olci_"}
SENSORS = ["hyp", "msi", "olci", "msi_strict", "olci_strict"]
# Strict band sets (Amendment 1 item 1 sensitivity; AUDIT_3 A3-4): MSI B1-B4, OLCI Oa2-Oa10. They have no
# red-edge band, so the NDCI ratio feature and the NDCI empirical baseline are not available for them.
STRICT_SENSORS = {"msi_strict": "msi", "olci_strict": "olci"}

BAND_ROLES = {
    "hyp": {"b443": "hyp_445", "b490": "hyp_490", "b510": "hyp_510", "g560": "hyp_560",
            "r665": "hyp_665", "re705": "hyp_705"},
    "msi": {"b443": "msi_B1", "b490": "msi_B2", "b510": None, "g560": "msi_B3",
            "r665": "msi_B4", "re705": "msi_B5"},
    "olci": {"b443": "olci_Oa3", "b490": "olci_Oa4", "b510": "olci_Oa5", "g560": "olci_Oa6",
             "r665": "olci_Oa8", "re705": "olci_Oa11"},
}
BAND_ROLES["msi_strict"] = {**BAND_ROLES["msi"], "re705": None}
BAND_ROLES["olci_strict"] = {**BAND_ROLES["olci"], "re705": None}


def feature_columns(sensor: str) -> list[str]:
    if sensor == "hyp":
        return [f"hyp_{w}" for w in config.HYP_CENTRES]
    if sensor == "msi":
        return [f"msi_{b}" for b in config.MSI_BANDS]
    if sensor == "olci":
        return [f"olci_{b}" for b in config.OLCI_BANDS]
    if sensor == "msi_strict":
        return [f"msi_{b}" for b in config.MSI_STRICT_BANDS]
    if sensor == "olci_strict":
        return [f"olci_{b}" for b in config.OLCI_STRICT_BANDS]
    raise ValueError(sensor)


def complete_column(sensor: str) -> str | None:
    """Stored completeness flag in gloria.parquet; None for strict sets (completeness = all strict bands present)."""
    return None if sensor in STRICT_SENSORS else f"complete_{sensor}"


def complete_mask(g: pd.DataFrame, sensor: str) -> np.ndarray:
    col = complete_column(sensor)
    if col is None:
        return g[feature_columns(sensor)].notna().all(axis=1).to_numpy()
    return g[col].astype(bool).to_numpy()


def log_bands(X: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(X, LOG_FLOOR))


def ratio_features(df: pd.DataFrame, sensor: str) -> pd.DataFrame:
    """Band ratios used by the empirical baselines (added to LightGBM inputs; dossier 5.1)."""
    r = BAND_ROLES[sensor]
    fl = lambda c: np.maximum(df[c].to_numpy(float), LOG_FLOOR)
    blues = [r[k] for k in ("b443", "b490", "b510") if r[k]]
    mbr = np.max(np.stack([fl(c) for c in blues], 1), 1) / fl(r["g560"])
    out = {"x_log_mbr": np.log10(mbr)}
    if r["re705"] is not None:  # strict band sets have no red-edge band: no NDCI feature
        re, red = df[r["re705"]].to_numpy(float), df[r["r665"]].to_numpy(float)
        denom = re + red
        out["x_ndci"] = np.where(np.abs(denom) > 1e-12, (re - red) / np.where(denom == 0, 1, denom), 0.0)
    out["x_log_g_r"] = np.log10(fl(r["g560"]) / fl(r["r665"]))
    out["x_log_b_g"] = np.log10(fl(r["b490"]) / fl(r["g560"]))
    return pd.DataFrame(out, index=df.index)
