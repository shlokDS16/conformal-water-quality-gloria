"""Fig. A.2: water-body size distribution (single column, appendix).

Data: data/processed/groups_<target>.parquet, column n_samples (samples per wb_group in the
primary population). (a) empirical CDF of samples per water body, log x; (b) cumulative share
of samples against the share of water bodies ranked from largest to smallest.
Run: python -m src.figures.figA2_wbsize
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from . import style as S

STYLE = dict(zip(S.TARGETS, [("-", S.CATEGORICAL[0]), ("--", S.CATEGORICAL[1]), ("-.", S.CATEGORICAL[2]),
                             (":", S.CATEGORICAL[3])]))


def main() -> None:
    S.apply()
    pop = S.load_gloria(list(S.TARGET_POP.values()))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(S.SINGLE_COL, 3.9))
    fig.subplots_adjust(left=0.17, right=0.95, top=0.94, bottom=0.11, hspace=0.5)
    for t in S.TARGETS:
        g = pd.read_parquet(S.DATA / "processed" / f"groups_{t}.parquet")
        n = np.sort(g["n_samples"].to_numpy())
        assert n.sum() == int(pop[S.TARGET_POP[t]].sum()), t
        ls, col = STYLE[t]
        a1.step(n, np.arange(1, len(n) + 1) / len(n), where="post", linestyle=ls, color=col,
                label=f"{S.TARGET_SHORT[t]} ({len(n)})")
        desc = n[::-1]
        a2.plot(np.r_[0, np.arange(1, len(n) + 1) / len(n)], np.r_[0, np.cumsum(desc) / desc.sum()], linestyle=ls,
                color=col)
    a1.set_xscale("log")
    a1.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    a1.set_xlabel("Samples per water body")
    a1.set_ylabel("Share of water bodies")
    a1.set_ylim(0, 1.02)
    a1.grid(True)
    a1.legend(loc="lower right", title="Target (water bodies)", handlelength=2.4)
    S.panel_letter(a1, "a", x=-0.04, y=1.03)
    a2.plot([0, 1], [0, 1], color=S.GREY, linewidth=0.8, linestyle=(0, (1, 2)))
    a2.set_xlabel("Share of water bodies, largest first")
    a2.set_ylabel("Share of samples")
    a2.set_xlim(0, 1)
    a2.set_ylim(0, 1.02)
    a2.grid(True)
    a2.text(0.62, 0.50, "equal sizes", fontsize=S.SIZE_TICK, color=S.GREY, rotation=0, ha="left")
    S.panel_letter(a2, "b", x=-0.04, y=1.03)
    S.save(fig, "figA2_wbsize")


if __name__ == "__main__":
    main()
