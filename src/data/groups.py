"""Water-body, replicate and contributor grouping keys.

Nodes and links
- Node = normalised Site_name (lower case, diacritics stripped, known typos fixed, trailing
  station numbers and compass suffixes stripped). A name used in more than one macro-region
  ("atlantic ocean": Bay of Biscay/Galicia vs US East Coast/Bahamas) becomes one node per region.
- Manual merges (AUDIT_1 L2 plus lead review of the candidate report) union nodes that are the
  same water body but fall outside the 2 km rule. Listed in MANUAL_MERGES and written to
  data/interim/wb_overrides.csv.
- Spatial link: two nodes are unioned when any two of their samples are within R km (haversine).
- Sample-identity links (applied to every key): identical 8-decimal spectrum (spec_hash); same
  latitude, longitude and datetime (replicates, AUDIT_1 L5); spectral near-duplicates with RMS
  difference of log10 Rrs over 400-750 nm < 0.005 AND identical target values (AUDIT_1 L3);
  same (lat, lon, datetime, targets) across Dataset_ID (AUDIT_1 L4); shared LIMNADES_ID or
  LIMNADES_UID (checked: shared values are within one dataset and one site).
- SeaBASS_ID is a cruise-level DOI (24 values over 1,093 rows), so it is never used.

Keys
- wb_group (primary, 2 km), wb_5km, wb_raw (raw Site_name nodes, no spatial link, no manual merge).
- iid_unit: components of the sample-identity links only (unit of the random protocol).
- contrib_group: components of the bipartite Dataset_ID <-> wb_group graph (AUDIT_1 L1).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree, NearestNeighbors

EARTH_RADIUS_KM = 6371.0088
NEAR_DUP_RMS = 0.005

TYPO_FIXES = {
    "gulf of mexcio": "gulf of mexico",
    "loosdtrechtse": "loosdrechtse",
}
_STATION_SUFFIX = re.compile(r"\s*(#\s*\d+|nr\.?\s*\d+)$")
_COMPASS_SUFFIX = re.compile(r"\s+(oost|west|noord|zuid|north|south|east)$")

# Manual merges of normalised names: (names, reason, source)
MANUAL_MERGES = [
    (["ijsselmeer", "ijsselmeer de oude zeug"], "same lake; station 24.9 km from main samples", "AUDIT_1 L2"),
    (["aransas bay, tx", "aransas pass, tx", "aransas channel, tx", "lydia ann channel, tx"],
     "connected Aransas estuary channels, 8-18 km apart", "AUDIT_1 L2"),
    (["minnetonka halstead's bay", "minnetonka maxwell bay", "minnetonka north arm", "minnetonka stubbs bay",
      "minnetonka upper lake", "minnetonka west arm", "minnetonka west upper lake"],
     "bays and arms of Lake Minnetonka, 4-7 km apart", "AUDIT_1 L2"),
    (["amsterdam-rhine canal", "amsterdam-rijn kanaal weesp"], "same canal, English and Dutch names, 14.7 km", "AUDIT_1 L2"),
]
# After review by eye of data/interim/wb_merge_candidates.csv (408 pairs within 25 km, 14 token-only pairs).
# Criterion: the names denote the same named water body (a part, arm or station of one lake, sea or fjord).
# Distinct named lakes, tributaries or connected bays were NOT merged.
EXTRA_MERGES = [
    (["gulf of mexico, la", "northern gulf of mexico"], "same sea (Louisiana shelf), closest samples 2.4 km", "pipeline review of candidates"),
    (["ijmeer durgerdam", "markermeer de hemmelanden"], "IJmeer is the south-western part of Markermeer, 9.9 km", "pipeline review of candidates"),
    (["sognefjord", "sognefjord/north sea"], "same fjord (inner fjord and fjord mouth)", "pipeline review of candidates"),
    (["south china sea", "south china sea (bornean coast)"], "same named sea, same macro-region (1 Bornean sample)", "pipeline review of candidates"),
]


def normalise_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"\s+", " ", s.lower()).strip()
    for bad, good in TYPO_FIXES.items():
        s = s.replace(bad, good)
    s = _STATION_SUFFIX.sub("", s)
    s = _COMPASS_SUFFIX.sub("", s)
    return s.strip()


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, a: int) -> int:
        p = self.parent
        root = a
        while p[root] != root:
            root = p[root]
        while p[a] != root:
            p[a], a = root, p[a]
        return int(root)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def labels(self, n: int) -> np.ndarray:
        return np.array([self.find(i) for i in range(n)])


def spectrum_hash(rrs: pd.DataFrame) -> pd.Series:
    vals = rrs.drop(columns="GLORIA_ID").to_numpy(dtype=float)
    out = []
    for row in vals:
        txt = ",".join("nan" if not np.isfinite(v) else f"{v:.8f}" for v in row)
        out.append(hashlib.sha1(txt.encode()).hexdigest()[:16])
    return pd.Series(out, index=rrs.index, name="spec_hash")


def _pairs_from_codes(codes: np.ndarray) -> list[tuple[int, int]]:
    s = pd.Series(codes)
    s = s[s >= 0]
    pairs = []
    for idx in s.groupby(s).indices.values():
        idx = s.index.to_numpy()[idx]
        pairs.extend((int(idx[0]), int(j)) for j in idx[1:])
    return pairs


def identity_links(df: pd.DataFrame, rrs: pd.DataFrame, spec_hash: pd.Series, targets: list[str]) -> tuple[list, dict]:
    """Row pairs that represent the same sample or a same-time replicate."""
    info: dict = {}
    pairs: list[tuple[int, int]] = []
    p = _pairs_from_codes(pd.factorize(spec_hash)[0])
    info["pairs_spec_hash"] = len(p)
    info["rows_in_dup_spectra"] = int(spec_hash.duplicated(keep=False).sum())
    pairs += p

    ok = df["Latitude"].notna() & df["Longitude"].notna() & df["Date_Time_UTC"].notna()
    key = df["Latitude"].round(5).astype(str) + "|" + df["Longitude"].round(5).astype(str) + "|" + df["Date_Time_UTC"].astype(str)
    codes = np.where(ok, pd.factorize(key)[0], -1)
    p = _pairs_from_codes(codes)
    info["pairs_same_lat_lon_datetime"] = len(p)
    info["rows_in_same_lat_lon_datetime"] = int((pd.Series(codes)[codes >= 0].duplicated(keep=False)).sum())
    kd = pd.DataFrame({"k": key[ok], "d": df.loc[ok, "Dataset_ID"]})
    info["same_lat_lon_datetime_keys_across_datasets"] = int((kd.groupby("k")["d"].nunique() > 1).sum())
    pairs += p

    tkey = key + "|" + df[targets].round(6).astype(str).agg("|".join, axis=1)
    kd = pd.DataFrame({"k": tkey[ok], "d": df.loc[ok, "Dataset_ID"]})
    cross = kd.groupby("k")["d"].nunique()
    info["lat_lon_datetime_target_dups_across_datasets"] = int((cross > 1).sum())
    for k in cross[cross > 1].index:
        idx = kd.index[kd["k"] == k].to_numpy()
        pairs += [(int(idx[0]), int(j)) for j in idx[1:]]

    # spectral near-duplicates: log10 Rrs 400-750 nm, complete and positive
    cols = [f"Rrs_{w}" for w in range(400, 751)]
    X = rrs[cols].to_numpy(dtype=float)
    good = np.isfinite(X).all(axis=1) & (X > 0).all(axis=1)
    info["near_dup_rows_eligible"] = int(good.sum())
    L = np.log10(X[good])
    gidx = np.flatnonzero(good)
    nn = NearestNeighbors(radius=NEAR_DUP_RMS * np.sqrt(L.shape[1]), algorithm="ball_tree").fit(L)
    _, nbrs = nn.radius_neighbors(L)
    tv = df[targets].round(10).astype(str).agg("|".join, axis=1).to_numpy()
    n_close, n_close_same_target = 0, 0
    rows_close = set()
    for i, js in enumerate(nbrs):
        for j in js:
            if j <= i:
                continue
            n_close += 1
            a, b = int(gidx[i]), int(gidx[j])
            rows_close.update((a, b))
            if tv[a] == tv[b]:
                n_close_same_target += 1
                pairs.append((a, b))
    info["near_dup_pairs_rms_lt_0.005"] = n_close
    info["near_dup_rows_rms_lt_0.005"] = len(rows_close)
    info["near_dup_pairs_same_targets_unioned"] = n_close_same_target

    for col in ["LIMNADES_ID", "LIMNADES_UID"]:
        vals = df[col].astype("string")
        codes = pd.factorize(vals, use_na_sentinel=True)[0]
        p = _pairs_from_codes(codes)
        info[f"pairs_{col}"] = len(p)
        shared = vals[vals.notna() & vals.duplicated(keep=False)]
        info[f"{col}_shared_across_datasets"] = int((df.loc[shared.index].groupby(col)["Dataset_ID"].nunique() > 1).sum())
        pairs += p
    sb = df["SeaBASS_ID"].astype("string")
    info["SeaBASS_ID_unique"] = int(sb.nunique())
    info["SeaBASS_ID_rows"] = int(sb.notna().sum())
    return pairs, info


def _node_components(nodes: pd.Series, lat: np.ndarray, lon: np.ndarray, radius_km: float | None,
                     row_pairs: list, node_merges: list[list[str]]) -> pd.Series:
    codes, uniq = pd.factorize(nodes, sort=True)
    uf = UnionFind(len(uniq))
    pos = {n: i for i, n in enumerate(uniq)}
    for group in node_merges:
        present = [pos[n] for n in group if n in pos]
        for b in present[1:]:
            uf.union(present[0], b)
    if radius_km is not None:
        ok = np.isfinite(lat) & np.isfinite(lon)
        X = np.radians(np.column_stack([lat[ok], lon[ok]]))
        nodes_ok = codes[ok]
        tree = BallTree(X, metric="haversine")
        for i, js in enumerate(tree.query_radius(X, r=radius_km / EARTH_RADIUS_KM)):
            for b in np.unique(nodes_ok[js]):
                uf.union(nodes_ok[i], b)
    for a, b in row_pairs:
        uf.union(codes[a], codes[b])
    comp = np.array([uf.find(c) for c in codes])
    first = pd.Series(nodes.to_numpy()).groupby(comp).transform("min")  # unique label per component
    return pd.Series(first.to_numpy(), index=nodes.index)


def build_groups(df: pd.DataFrame, rrs: pd.DataFrame, targets: list[str]) -> tuple[pd.DataFrame, dict]:
    """df and rrs must be row-aligned; df needs a `region` column."""
    assert (df["GLORIA_ID"].to_numpy() == rrs["GLORIA_ID"].to_numpy()).all()
    df = df.reset_index(drop=True)
    rrs = rrs.reset_index(drop=True)
    out = pd.DataFrame({"GLORIA_ID": df["GLORIA_ID"].to_numpy()})
    out["site_norm"] = df["Site_name"].map(normalise_name)
    out["spec_hash"] = spectrum_hash(rrs)
    lat = df["Latitude"].to_numpy(dtype=float)
    lon = df["Longitude"].to_numpy(dtype=float)
    pairs, info = identity_links(df, rrs, out["spec_hash"], targets)

    def region_nodes(names: pd.Series) -> tuple[pd.Series, list[str]]:
        nreg = df.groupby(names.to_numpy())["region"].nunique()
        multi = sorted(nreg[nreg > 1].index)
        return pd.Series([f"{n} [{r}]" if n in multi else n for n, r in zip(names, df["region"])]), multi

    norm_nodes, split_norm = region_nodes(out["site_norm"])
    raw_nodes, split_raw = region_nodes(df["Site_name"])
    info["names_split_by_region"] = {"normalised": split_norm, "raw": split_raw}
    merges = [m[0] for m in MANUAL_MERGES + EXTRA_MERGES]
    missing = sorted({n for g in merges for n in g} - set(out["site_norm"]))
    assert not missing, f"manual merge names not found: {missing}"
    out["wb_node"] = norm_nodes.to_numpy()
    out["wb_group"] = "wb2:" + _node_components(norm_nodes, lat, lon, 2.0, pairs, merges)
    out["wb_5km"] = "wb5:" + _node_components(norm_nodes, lat, lon, 5.0, pairs, merges)
    out["wb_raw"] = "raw:" + _node_components(raw_nodes, lat, lon, None, pairs, [])

    # iid_unit: identity links only
    uf = UnionFind(len(df))
    for a, b in pairs:
        uf.union(a, b)
    gid = pd.Series(df["GLORIA_ID"].to_numpy())
    out["iid_unit"] = "iid:" + gid.groupby(uf.labels(len(df))).transform("min").to_numpy()  # order-invariant label

    # contrib_group: bipartite Dataset_ID <-> wb_group components
    ds_codes, ds_u = pd.factorize(df["Dataset_ID"], sort=True)
    wb_codes, wb_u = pd.factorize(out["wb_group"], sort=True)
    uf = UnionFind(len(ds_u) + len(wb_u))
    for a, b in zip(ds_codes, wb_codes):
        uf.union(a, len(ds_u) + b)
    comp = np.array([uf.find(a) for a in ds_codes])
    out["contrib_group"] = "cg:" + pd.Series(df["Dataset_ID"].to_numpy()).groupby(comp).transform("min").to_numpy()

    # Diagnostics for the dossier comparison (names only, no region split, no merges, no links)
    pure = out["site_norm"]
    info["n_raw_names"] = int(df["Site_name"].nunique())
    info["n_norm_names"] = int(pure.nunique())
    for km in (0.5, 2.0, 5.0, 10.0):
        info[f"dossier_def_{km}km"] = int(_node_components(pure, lat, lon, km, [], []).nunique())
    fin = np.isfinite(lat)
    info["dossier_def_2.0km_coord_rows_only"] = int(_node_components(pure[fin].reset_index(drop=True), lat[fin], lon[fin], 2.0, [], []).nunique())
    info["dossier_def_5.0km_coord_rows_only"] = int(_node_components(pure[fin].reset_index(drop=True), lat[fin], lon[fin], 5.0, [], []).nunique())
    info["stage_2km_plus_region_split"] = int(_node_components(norm_nodes, lat, lon, 2.0, [], []).nunique())
    info["stage_2km_plus_region_split_plus_merges"] = int(_node_components(norm_nodes, lat, lon, 2.0, [], merges).nunique())
    for k in ["wb_group", "wb_5km", "wb_raw", "iid_unit", "contrib_group"]:
        info[f"n_{k}"] = int(out[k].nunique())
    cg = out.groupby("contrib_group").size().sort_values(ascending=False)
    info["contrib_group_largest"] = {k: int(v) for k, v in cg.head(5).items()}
    info["contrib_group_datasets"] = {k: int(v) for k, v in df.groupby(out["contrib_group"])["Dataset_ID"].nunique().sort_values(ascending=False).head(5).items()}
    info["wb_groups_spanning_datasets"] = int((df.groupby(out["wb_group"])["Dataset_ID"].nunique() > 1).sum())
    return out, info


def merge_candidates(df: pd.DataFrame, keys: pd.DataFrame, max_km: float = 25.0) -> pd.DataFrame:
    """Pairs of distinct wb_groups that are within max_km (closest samples) or share a distinctive name token."""
    lat = df["Latitude"].to_numpy(float)
    lon = df["Longitude"].to_numpy(float)
    ok = np.isfinite(lat)
    X = np.radians(np.column_stack([lat[ok], lon[ok]]))
    g = keys["wb_group"].to_numpy()[ok]
    tree = BallTree(X, metric="haversine")
    dist, idx = {}, tree.query_radius(X, r=max_km / EARTH_RADIUS_KM, return_distance=True)
    for i, (js, ds) in enumerate(zip(*idx)):
        for j, d in zip(js, ds):
            if g[i] < g[j]:
                k = (g[i], g[j])
                dist[k] = min(dist.get(k, 1e9), d * EARTH_RADIUS_KM)
    stop = {"lake", "river", "bay", "sea", "of", "the", "ocean", "de", "la", "reservoir", "gulf", "creek", "sound",
            "channel", "lagoon", "island", "harbor", "pond", "mexico", "atlantic", "pacific", "fl", "tx", "ga", "sc",
            "md", "ms", "va", "de", "nc", "ne", "ny", "nj", "ok", "ia", "in", "la", "ca", "plas", "plassen", "and",
            "del", "en", "or", "north", "south", "east", "west", "upper", "lower", "inshore", "offshore", "lac",
            "loch", "embalse", "lago", "dam", "water", "waterway", "intracoastal", "bayou", "arm", "point", "st.",
            "[europe]", "[north", "america]"}
    tok = keys.groupby("wb_group")["site_norm"].apply(
        lambda s: {t for n in s for t in re.split(r"[\s,/()\-]+", n) if len(t) > 3 and t not in stop})
    names = keys.groupby("wb_group")["site_norm"].apply(lambda s: "; ".join(sorted(set(s))))
    inv: dict = {}
    for grp, ts in tok.items():
        for t in ts:
            inv.setdefault(t, set()).add(grp)
    shared = {}
    for t, gs in inv.items():
        gs = sorted(gs)
        for a in range(len(gs)):
            for b in range(a + 1, len(gs)):
                shared.setdefault((gs[a], gs[b]), set()).add(t)
    keys_all = set(dist) | set(shared)
    rows = [{"group_a": a, "group_b": b, "names_a": names[a], "names_b": names[b],
             "min_km": round(dist[(a, b)], 2) if (a, b) in dist else np.nan,
             "shared_tokens": " ".join(sorted(shared.get((a, b), [])))} for a, b in sorted(keys_all)]
    return pd.DataFrame(rows).sort_values(["min_km", "group_a"], na_position="last").reset_index(drop=True)


def group_table(df: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    merged_names = {n: f"manual merge ({src}): {reason}" for names, reason, src in MANUAL_MERGES + EXTRA_MERGES for n in names}
    t = pd.DataFrame({
        "raw_name": df["Site_name"].to_numpy(), "normalised_name": keys["site_norm"].to_numpy(),
        "wb_node": keys["wb_node"].to_numpy(), "dataset": df["Dataset_ID"].to_numpy(), "country": df["Country"].to_numpy(),
        "lat": df["Latitude"].to_numpy(), "lon": df["Longitude"].to_numpy(), "wb_group": keys["wb_group"].to_numpy(),
        "wb_5km": keys["wb_5km"].to_numpy(), "wb_raw": keys["wb_raw"].to_numpy(), "contrib_group": keys["contrib_group"].to_numpy(),
    })
    g = (t.groupby(["raw_name", "normalised_name", "wb_node", "dataset", "country", "wb_group", "wb_5km", "wb_raw", "contrib_group"], dropna=False)
         .agg(n=("lat", "size"), lat_centroid=("lat", "mean"), lon_centroid=("lon", "mean")).reset_index())
    notes = []
    for _, r in g.iterrows():
        parts = []
        if r["normalised_name"] != r["raw_name"].lower():
            parts.append("name normalised")
        if r["wb_node"] != r["normalised_name"]:
            parts.append("name split by macro-region")
        if r["normalised_name"] in merged_names:
            parts.append(merged_names[r["normalised_name"]])
        notes.append("; ".join(parts))
    g["override"] = notes
    cols = ["raw_name", "normalised_name", "wb_node", "dataset", "country", "n", "lat_centroid", "lon_centroid",
            "wb_group", "wb_5km", "wb_raw", "contrib_group", "override"]
    return g[cols].sort_values(["wb_group", "normalised_name", "raw_name", "dataset"]).reset_index(drop=True)


def overrides_table() -> pd.DataFrame:
    rows = [{"rule": "merge", "names": " | ".join(n), "reason": r, "source": s} for n, r, s in MANUAL_MERGES + EXTRA_MERGES]
    rows.append({"rule": "split by macro-region", "names": "atlantic ocean",
                 "reason": "one raw name covers Bay of Biscay/Galicia (Europe, 14 rows) and US East Coast/Bahamas (North America, 48 rows)",
                 "source": "AUDIT_1 L2"})
    rows.append({"rule": "region override", "names": "Guiana", "reason": "Country = France but French Guiana lies in South America",
                 "source": "pipeline review"})
    return pd.DataFrame(rows)


def majority_by_group(values: pd.Series, groups: pd.Series) -> tuple[pd.Series, int]:
    d = pd.DataFrame({"v": values.to_numpy(), "g": groups.to_numpy()}).dropna(subset=["v"])
    cnt = d.groupby(["g", "v"]).size().reset_index(name="n").sort_values(["g", "n", "v"], ascending=[True, False, True])
    conflicts = int((cnt.groupby("g").size() > 1).sum())
    top = cnt.drop_duplicates("g").set_index("g")["v"]
    return pd.Series(groups.map(top).to_numpy(), index=values.index), conflicts
