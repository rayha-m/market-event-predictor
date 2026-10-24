from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Values copied from the README tables (docs/test_results.txt and docs/backtest_results.txt).
direction = [  # label, point, low, high  (AUC-based, 0 = chance)
    ("Text minus ticker-only\nvalidation 2022-23 (examined)", 0.0175, 0.0088, 0.0256),
    ("Text minus ticker-only\ntest 2024-26", 0.0025, -0.0074, 0.0122),
    ("Text AUC minus 0.5\ntest 2024-26", 0.0022, -0.0121, 0.0176),
    ("Ticker-only AUC minus 0.5\ntest 2024-26", -0.0003, -0.0165, 0.0162),
    ("Text AUC minus 0.5, beta-adjusted\ntest 2024-26", 0.0012, -0.0129, 0.0164),
]
backtest = [  # basis points per trade
    ("2016-2023 gross", 6.9, -0.2, 14.1),
    ("2016-2023 net (10 bps/side)", -33.1, -40.2, -25.9),
    ("2024-2026 gross", -5.6, -20.3, 8.7),
    ("2024-2026 net (10 bps/side)", -45.6, -60.3, -31.3),
]


def panel(ax, rows, title, xlabel):
    for i, (label, pt, lo, hi) in enumerate(rows):
        color = "tab:red" if (lo > 0 or hi < 0) else "tab:gray"
        ax.errorbar(pt, i, xerr=[[pt - lo], [hi - pt]], fmt="o", color=color, capsize=4)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(axis="x", alpha=0.3)


fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.2))
panel(a, direction, "Direction: nothing beyond chance on the test set", "AUC difference, 95% week-block interval")
panel(b, backtest, "Backtest: no edge before costs, losses after", "Return per trade (bps), 95% interval")
fig.text(0.5, 0.005, "Red = interval excludes zero. Dashed line = no effect.", ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = Path("docs/results.png")
fig.savefig(out, dpi=150)
print("saved", out)