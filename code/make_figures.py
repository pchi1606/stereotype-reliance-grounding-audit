"""Generate paper figures from outputs/results.json + generations.

Fig 1: mean Wasserstein distance by arm (H2).
Fig 2: entropy-ratio distributions by arm (H3 / flattening).
Fig 3: k-NN separability by arm vs human baseline (H4).
Fig 4: per-item Wasserstein heatmap (arms x items, averaged over countries).
All figures regenerated from real data (test 12).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "code")
from metrics import as_dist, wasserstein_ordinal, entropy_ratio
from analyze import opt_index

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
ARMS = ["label", "retrieval", "forecast"]
NAMES = {"label": "Bare label", "retrieval": "Retrieval-grounded",
         "forecast": "Forecasting", "neutral": "Neutral"}
COLORS = {"label": "#c0392b", "retrieval": "#2471a3", "forecast": "#229954"}

items = {m["item_id"]: m for m in json.load(open("data/items.json"))}
df = pd.DataFrame([json.loads(l) for l in open("outputs/generations.jsonl")])
d = df[df["run_id"].str.startswith("M-")]
countries = sorted(c for c in d["country"].dropna().unique() if c)
item_ids = sorted(items)

def cell_dist(arm, c, i):
    g = d[(d["arm"] == arm) & (d["country"] == c) & (d["item_id"] == i)]
    k = len(items[i]["options"])
    counts = np.zeros(k)
    for o in g["option"]:
        j = opt_index(o, k)
        if j is not None: counts[j] += 1
    return as_dist(counts, k)

W = {(a, c, i): wasserstein_ordinal(cell_dist(a, c, i),
                                    np.array(items[i]["human_dist"][c]))
     for a in ARMS for c in countries for i in item_ids}
R = {(a, c, i): entropy_ratio(cell_dist(a, c, i),
                              np.array(items[i]["human_dist"][c]))
     for a in ARMS for c in countries for i in item_ids}

# Fig 1: mean Wasserstein by arm with 95% CI
fig, ax = plt.subplots(figsize=(5, 3.2))
means, cis = [], []
for a in ARMS:
    v = np.array([W[(a, c, i)] for c in countries for i in item_ids])
    means.append(v.mean()); cis.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
ax.bar([NAMES[a] for a in ARMS], means, yerr=cis, capsize=4,
       color=[COLORS[a] for a in ARMS])
ax.set_ylabel("Mean Wasserstein distance to human distribution")
ax.set_title("Distributional fidelity by grounding arm (lower is better)")
fig.tight_layout(); fig.savefig("paper/figures/fig1_wasserstein.png")

# Fig 2: entropy ratio distributions
fig, ax = plt.subplots(figsize=(5, 3.2))
data = [[R[(a, c, i)] for c in countries for i in item_ids] for a in ARMS]
bp = ax.boxplot(data, labels=[NAMES[a] for a in ARMS], patch_artist=True)
for p, a in zip(bp["boxes"], ARMS): p.set_facecolor(COLORS[a])
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_ylabel("Entropy ratio model/human (1 = no flattening)")
ax.set_title("Within-cell dispersion vs human dispersion")
fig.tight_layout(); fig.savefig("paper/figures/fig2_entropy_ratio.png")

# Fig 3: k-NN separability (from results.json)
res = json.load(open("outputs/results.json"))
fig, ax = plt.subplots(figsize=(5, 3.2))
arms4 = ARMS + ["human"]
vals = [res["knn_separability"].get(a, res["knn_separability_human"])
        if a != "human" else res["knn_separability_human"] for a in arms4]
ax.bar(["Bare label", "Retrieval", "Forecasting", "Human\nbaseline"], vals,
       color=["#c0392b", "#2471a3", "#229954", "#7f8c8d"])
ax.set_ylabel("1-NN leave-one-out country separability (per-generation profiles)")
ax.set_title("Country separability: 108 profiles per arm (chance = 1/9)")
ax.axhline(1/9, color="k", ls="--", lw=1, label="chance")
ax.set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig("paper/figures/fig3_separability.png")

# Fig 4: per-item mean Wasserstein heatmap
mat = np.array([[np.mean([W[(a, c, i)] for c in countries]) for i in item_ids]
                for a in ARMS])
fig, ax = plt.subplots(figsize=(7, 2.6))
im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
ax.set_yticks(range(3)); ax.set_yticklabels([NAMES[a] for a in ARMS])
ax.set_xticks(range(len(item_ids)))
ax.set_xticklabels([f"{i}\n{items[i]['prompt_q'][:28]}..." for i in item_ids],
                   rotation=45, ha="right", fontsize=7)
ax.set_title("Mean Wasserstein distance per item and arm")
fig.colorbar(im, ax=ax, label="Wasserstein")
fig.tight_layout(); fig.savefig("paper/figures/fig4_item_heatmap.png")
print("figures written")
