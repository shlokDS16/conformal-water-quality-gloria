"""Project-wide constants: seeds, targets, feature sets. Single source of truth."""
from __future__ import annotations

import zlib

# Seeds (AUDIT_1 F3 / lead rule e). Split seed s; model seed 1000+s; ensemble member seed 1000+10*s+m.
SEEDS = range(20)
SEEDS_RANDOM = list(range(20))
SEEDS_WATERBODY = list(range(20))
SEEDS_CONTRIBUTOR = list(range(10))
SEEDS_REGION = list(range(5))


def split_seed(protocol: str, seed: int) -> int:
    """Deterministic 31-bit RNG seed from (protocol, seed); independent of PYTHONHASHSEED."""
    return zlib.crc32(f"{protocol}|{seed}".encode()) & 0x7FFFFFFF


def model_seed(seed: int) -> int:
    return 1000 + seed


def member_seed(seed: int, member: int) -> int:
    return 1000 + 10 * seed + member


# MDN member seeds per purpose (AUDIT_3 A3-5 fix, tag core_v2 onwards). Collision-free by construction:
#   member_seed(s, m) = 1000 + 10 s + m lies in [1000, 10000) for s in [0, 899], m in [0, 9] and is
#   injective in (s, m). Each purpose adds a disjoint offset block:
#     full ensemble    : member_seed                                  in [1000, 10000)
#     recal 80 % train : 500000 + member_seed                         in [501000, 510000)
#     gauss-CV fold k  : 600000 + 10000 k + member_seed, k in [0, 9]  in [601000, 700000), one block per k
# so (purpose, fold, s, m) -> seed is injective. The former rule 600000 + 100 k + member_seed collided
# for s >= 10 (AUDIT_3 A3-5).
MDN_RECAL_OFFSET = 500_000
MDN_CV_OFFSET = 600_000
MDN_CV_FOLD_STRIDE = 10_000
MAX_SPLIT_SEED = 899
MAX_MEMBERS = 10
MAX_CV_FOLDS = 10


def _check_member(seed: int, member: int) -> None:
    if not (0 <= seed <= MAX_SPLIT_SEED and 0 <= member < MAX_MEMBERS):
        raise ValueError(f"seed {seed} or member {member} outside the collision-free range")


def mdn_member_seed(purpose: str, seed: int, member: int, fold: int = 0) -> int:
    """MDN member seed for purpose in {'full', 'r80', 'cv'}; fold is the gauss-CV fold (cv only)."""
    _check_member(seed, member)
    base = member_seed(seed, member)
    if purpose == "full":
        return base
    if purpose == "r80":
        return MDN_RECAL_OFFSET + base
    if purpose == "cv":
        if not 0 <= fold < MAX_CV_FOLDS:
            raise ValueError(f"cv fold {fold} outside [0, {MAX_CV_FOLDS})")
        return MDN_CV_OFFSET + MDN_CV_FOLD_STRIDE * fold + base
    raise ValueError(purpose)


RAW_TARGETS = ["Chla", "TSS", "aCDOM440", "Secchi_depth"]
# Split populations (boolean columns in gloria.parquet) per target.
PRIMARY_POPULATION = {
    "Chla": "chla_primary",
    "TSS": "tss_primary",
    "aCDOM440": "acdom_primary",
    "Secchi_depth": "secchi_primary",
}

# Default feature sets (see research/DATA_PIPELINE_LOG.md, implementation decisions).
MSI_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6"]
OLCI_BANDS = [f"Oa{i}" for i in range(2, 12)]
OLCI_EXCLUDED = ["Oa1", "Oa12"]  # 1%-of-peak support leaves GLORIA coverage for most spectra
# Strict band sets (Amendment 1 item 1 sensitivity): MSI B1-B4, OLCI Oa2-Oa10.
MSI_STRICT_BANDS = ["B1", "B2", "B3", "B4"]
OLCI_STRICT_BANDS = [f"Oa{i}" for i in range(2, 11)]
HYP_CENTRES = list(range(405, 746, 5))
HYP_EXCLUDED = [400, 750]
