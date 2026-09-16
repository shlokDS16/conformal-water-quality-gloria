"""Band simulation from GLORIA 1 nm Rrs (dossier 4.3).

Rrs_band = sum(Rrs(l) * SRF(l)) / sum(SRF(l)) over the 1 nm GLORIA wavelengths l where
SRF(l) >= 1% of the band peak. The band is missing for a sample if any GLORIA value in
that support is missing (no extrapolation). Truncating the SRF at 1% of peak keeps the
weights identical for every sample; the retained fraction of the SRF integral is logged
(>= 0.999 for all bands used). No solar-irradiance weighting (Rrs is already a ratio).

Hyperspectral features: 400-750 nm every 5 nm; each feature is the mean of the five 1 nm
values within +-2 nm of the centre, missing if any of the five is missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUPPORT_FRACTION = 0.01
HYP_HALF_WIDTH = 2


def rrs_matrix(rrs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cols = [c for c in rrs.columns if c.startswith("Rrs_")]
    wl = np.array([int(c.split("_")[1]) for c in cols])
    assert (np.diff(wl) == 1).all()
    return wl, rrs[cols].to_numpy(dtype=float)


def band_weights(srf: pd.Series, wl: np.ndarray) -> tuple[np.ndarray, bool]:
    """Weights on the GLORIA grid; second value False if the >=1% support leaves the GLORIA range."""
    s = srf / srf.max()
    support = s.index[s.to_numpy() >= SUPPORT_FRACTION].to_numpy()
    inside = support.min() >= wl.min() and support.max() <= wl.max()
    w = np.zeros(len(wl))
    idx = np.searchsorted(wl, support[(support >= wl.min()) & (support <= wl.max())])
    w[idx] = s.reindex(wl[idx]).to_numpy()
    return w, bool(inside)


def simulate(R: np.ndarray, wl: np.ndarray, srf: pd.DataFrame, bands: list[str], prefix: str) -> pd.DataFrame:
    out = {}
    for b in bands:
        w, inside = band_weights(srf[b], wl)
        if not inside:
            out[f"{prefix}{b}"] = np.full(R.shape[0], np.nan)
            continue
        m = w > 0
        sub = R[:, m]
        val = sub @ w[m] / w[m].sum()
        val[~np.isfinite(sub).all(axis=1)] = np.nan
        out[f"{prefix}{b}"] = val
    return pd.DataFrame(out)


def hyperspectral(R: np.ndarray, wl: np.ndarray, centres: list[int], prefix: str = "hyp_") -> pd.DataFrame:
    out = {}
    for c in centres:
        idx = np.searchsorted(wl, np.arange(c - HYP_HALF_WIDTH, c + HYP_HALF_WIDTH + 1))
        sub = R[:, idx]
        v = sub.mean(axis=1)
        v[~np.isfinite(sub).all(axis=1)] = np.nan
        out[f"{prefix}{c}"] = v
    return pd.DataFrame(out)


def irradiance_weighting_error(Lw: np.ndarray, Es: np.ndarray, wl: np.ndarray, srf: pd.DataFrame, bands: list[str]) -> pd.DataFrame:
    """AUDIT_1 D3: exact band Rrs = sum(Lw S)/sum(Es S) vs the unweighted sum((Lw/Es) S)/sum(S).

    Computed per band on rows where Lw and Es are complete and Es > 0 over the band support,
    so the comparison isolates the weighting approximation from GLORIA's own Rrs processing.
    """
    rows = []
    for b in bands:
        w, inside = band_weights(srf[b], wl)
        m = w > 0
        if not inside:
            continue
        lw, es = Lw[:, m], Es[:, m]
        ok = np.isfinite(lw).all(axis=1) & np.isfinite(es).all(axis=1) & (es > 0).all(axis=1)
        lw, es = lw[ok], es[ok]
        exact = lw @ w[m] / (es @ w[m])
        approx = (lw / es) @ w[m] / w[m].sum()
        good = np.abs(exact) > 1e-4
        rel = np.abs(approx[good] - exact[good]) / np.abs(exact[good])
        rows.append({"band": b, "n": int(good.sum()), "median_rel_diff": float(np.median(rel)) if len(rel) else np.nan,
                     "p95_rel_diff": float(np.percentile(rel, 95)) if len(rel) else np.nan})
    return pd.DataFrame(rows)


def flat_spectrum_check(srf: pd.DataFrame, bands: list[str], wl: np.ndarray, c: float = 0.0123) -> float:
    R = np.full((1, len(wl)), c)
    sim = simulate(R, wl, srf, bands, "x_").to_numpy()
    return float(np.nanmax(np.abs(sim - c)))


def smooth_b4_check(R: np.ndarray, wl: np.ndarray, msi_b4: np.ndarray, srf_b4: pd.Series) -> dict:
    """Among smooth spectra, fraction whose simulated B4 is within 5% of Rrs(665).

    Smooth: complete and positive over the B4 support, and the RMS residual of a straight-line
    fit of Rrs against wavelength over that support is < 2% of the mean Rrs there. (For a linear
    spectrum the SRF-weighted mean equals Rrs at the SRF centroid, 664.6 nm.) The same statistics
    are also returned for all complete spectra, where the 665 nm chlorophyll absorption feature
    makes B4 differ from Rrs(665) by design.
    """
    s = srf_b4 / srf_b4.max()
    sup = s.index[s.to_numpy() >= SUPPORT_FRACTION].to_numpy()
    sub = R[:, np.searchsorted(wl, sup)]
    complete = np.isfinite(sub).all(axis=1) & (np.nan_to_num(sub, nan=-1) > 0).all(axis=1)
    x = sup - sup.mean()
    X = np.column_stack([np.ones_like(x), x]).astype(float)
    Y = np.where(complete[:, None], sub, 0.0)
    coef, *_ = np.linalg.lstsq(X, Y.T, rcond=None)
    resid = Y - (X @ coef).T
    rel_rms = np.sqrt((resid ** 2).mean(axis=1)) / np.where(complete, Y.mean(axis=1), np.nan)
    smooth = complete & (rel_rms < 0.02)
    r665 = R[:, np.searchsorted(wl, 665)]
    rel = np.abs(msi_b4 - r665) / np.abs(r665)
    return {"n_smooth": int(smooth.sum()), "frac_within_5pct_smooth": float((rel[smooth] <= 0.05).mean()),
            "max_rel_diff_smooth": float(rel[smooth].max()), "median_rel_diff_smooth": float(np.median(rel[smooth])),
            "n_complete": int(complete.sum()), "frac_within_5pct_all_complete": float((rel[complete] <= 0.05).mean()),
            "median_rel_diff_all_complete": float(np.median(rel[complete]))}
