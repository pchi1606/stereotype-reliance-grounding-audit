"""Analysis: preregistered metrics + hypothesis tests (§§4-5).

Reads outputs/generations.jsonl + data/items.json. Writes outputs/results.json
and prints the hypothesis verdict table.
"""
import json, itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel, shapiro, norm
import sys
sys.path.insert(0, "code")
from metrics import (as_dist, wasserstein_ordinal, tvd, entropy_ratio,
                     country_profile_modal, knn_loo_accuracy,
                     knn_loo_accuracy_matrix, individuation,
                     exaggeration)

LETTERS = "ABCDEFGHIJ"

def load():
    items = {m["item_id"]: m for m in json.load(open("data/items.json"))}
    recs = [json.loads(l) for l in open("outputs/generations.jsonl")]
    df = pd.DataFrame(recs)
    return items, df

def opt_index(opt, k):
    o = (opt or "").strip().upper()[:1]
    return LETTERS.index(o) if o in LETTERS[:k] else None

def distributions(items, df, phase_prefix):
    """M[a][c][i] empirical distributions; H[c][i] human distributions."""
    M, H = {}, {}
    d = df[df["run_id"].str.startswith(phase_prefix)]
    for (arm, country, item_id), g in d.groupby(["arm", "country", "item_id"]):
        it = items[item_id]; k = len(it["options"])
        counts = np.zeros(k)
        for o in g["option"]:
            j = opt_index(o, k)
            if j is not None: counts[j] += 1
        M.setdefault(arm, {}).setdefault(country, {})[item_id] = as_dist(counts, k)
    for item_id, it in items.items():
        for c, h in it["human_dist"].items():
            H.setdefault(c, {})[item_id] = np.array(h)
    return M, H

def paired_test(x, y, name):
    x, y = np.asarray(x), np.asarray(y)
    d = x - y
    d = d[~np.isnan(d)]
    if len(d) < 8:
        return {"name": name, "n": len(d), "note": "too few pairs"}
    p_sw = shapiro(d).pvalue if len(d) <= 5000 else 0.0
    if p_sw < 0.05:
        stat, p = wilcoxon(d)
        meth = "wilcoxon"
    else:
        stat, p = ttest_rel(x, y)
        meth = "paired-t"
    sd = d.std(ddof=1)
    out = {"name": name, "n": len(d), "method": meth,
           "mean_diff": float(d.mean()),
           "cohens_d": float(d.mean() / sd) if sd > 0 else 0.0,
           "p": float(p), "p_shapiro": float(p_sw)}
    if meth == "wilcoxon":
        # rank-based effect size for a rank-based test: matched-pairs
        # rank-biserial r = Z / sqrt(N) (cohens_d is the standardized mean
        # of the paired differences, kept for reference)
        z = norm.ppf(1.0 - p / 2.0) * np.sign(d.mean())
        out["r_rank"] = float(z / np.sqrt(len(d)))
    return out

def main():
    items, df = load()
    M, H = distributions(items, df, "M-")
    arms = ["label", "retrieval", "forecast"]
    countries = sorted({c for a in M for c in M[a] if c})
    item_ids = sorted(items)

    # --- (a) Wasserstein per (arm, country, item); (b) entropy ratio
    W, R = {}, {}
    for a in arms:
        for c in countries:
            for i in item_ids:
                if c in M.get(a, {}) and i in M[a][c] and c in H and i in H[c]:
                    p, h = M[a][c][i], H[c][i]
                    W[(a, c, i)] = wasserstein_ordinal(p, h)
                    R[(a, c, i)] = entropy_ratio(p, h)
    cells = sorted({(c, i) for (a, c, i) in W})
    wArm = {a: [W[(a, c, i)] for (c, i) in cells] for a in arms}
    rArm = {a: [R[(a, c, i)] for (c, i) in cells] for a in arms}
    res = {"n_cells": len(cells), "countries": countries,
           "mean_wasserstein": {a: float(np.mean(wArm[a])) for a in arms},
           "mean_entropy_ratio": {a: float(np.mean(rArm[a])) for a in arms},
           "severe_flatten_frac": {a: float(np.mean(np.array(rArm[a]) < 0.5))
                                   for a in arms}}

    # H2: retrieval < label, forecast < label (Holm)
    h2 = [paired_test(wArm["label"], wArm["retrieval"], "H2a label-vs-retrieval"),
          paired_test(wArm["label"], wArm["forecast"], "H2b label-vs-forecast")]
    ps = sorted([h["p"] for h in h2])
    holm = [min(1.0, p * (2 - k)) for k, p in enumerate(ps)]
    for h, ph in zip(sorted(h2, key=lambda x: x["p"]), holm):
        h["p_holm"] = ph
        h["reject"] = (h["mean_diff"] > 0) and ph < 0.05
    res["H2"] = h2

    # H3: entropy ratio higher for retrieval/forecast than label
    h3 = [paired_test(rArm["retrieval"], rArm["label"], "H3a retrieval-vs-label"),
          paired_test(rArm["forecast"], rArm["label"], "H3b forecast-vs-label")]
    for h in h3:
        h["reject"] = (h["mean_diff"] > 0) and h["p"] < 0.05
    res["H3"] = h3

    # --- (c) caricature: k-NN separability over PER-GENERATION profiles.
    # One profile per generation (108 per arm: 9 countries x 12 generations),
    # each a 12-element vector of option indices (items in sorted order),
    # grouped by prompt_id. LOO 1-NN with Hamming distance; chance = 1/9.
    # (The preregistered modal-profile variant is degenerate here: with one
    # profile per country, LOO 1-NN can never predict the correct country, so
    # it is always 0.000 on every arm and the human baseline alike.)
    def gen_profiles(arm):
        d = df[(df["run_id"].str.startswith("M-")) & (df["arm"] == arm)]
        X, y = [], []
        for pid, g in d.groupby("prompt_id"):
            vec, ok = [], True
            for i in item_ids:
                gg = g[g["item_id"] == i]
                if len(gg) != 1:
                    ok = False; break
                j = opt_index(gg["option"].iloc[0], len(items[i]["options"]))
                if j is None:
                    ok = False; break
                vec.append(j)
            if ok:
                X.append(vec); y.append(g["country"].iloc[0])
        return np.array(X), np.array(y)
    knn = {}
    knn_n = {}
    for a in arms:
        Xa, ya = gen_profiles(a)
        knn[a] = float(knn_loo_accuracy_matrix(Xa, ya))
        knn_n[a] = int(len(Xa))
    # human baseline at the same granularity: 12 synthetic respondents per
    # country, each a vector of independent draws from that country's human
    # marginals (seeded RNG for reproducibility). This is NOT real human data;
    # it asks whether the human distributions themselves are country-separable.
    rng = np.random.default_rng(42)
    Xh, yh = [], []
    for c in countries:
        for s in range(12):
            vec = [int(rng.choice(len(items[i]["options"]),
                                 p=np.array(items[i]["human_dist"][c])))
                   for i in item_ids]
            Xh.append(vec); yh.append(c)
    Xh, yh = np.array(Xh), np.array(yh)
    knn_human = float(knn_loo_accuracy_matrix(Xh, yh))
    res["knn_separability"] = knn
    res["knn_separability_human"] = knn_human
    res["knn_chance"] = 1.0 / len(countries)
    res["knn_n_profiles"] = knn_n
    res["knn_human_method"] = ("12 synthetic respondents per country, "
                               "independent draws from human marginals, "
                               "rng seed 42")
    res["median_entropy_ratio"] = {a: float(np.median(rArm[a])) for a in arms}
    # --- modal country profiles (kept for exaggeration; individuation uses
    # per-generation profiles, which are non-degenerate)
    def profiles(arm):
        prof = {}
        for c in countries:
            vec = []
            for i in item_ids:
                if c in M.get(arm, {}) and i in M[arm][c]:
                    vec.append(M[arm][c][i])
            if len(vec) == len(item_ids):
                prof[c] = country_profile_modal(vec)
        return prof
    # human baseline profiles: GlobalOpinionQA modal option per (country,item)
    hprof = {c: np.array([int(np.argmax(items[i]["human_dist"][c]))
                          for i in item_ids]) for c in countries}
    mprof = {a: profiles(a) for a in arms}
    res["individuation"] = {}
    res["exaggeration"] = {}
    for a in arms:
        gen_prof = {}
        d = df[(df["run_id"].str.startswith("M-")) & (df["arm"] == a)]
        for c in countries:
            arr = []
            for i in item_ids:
                g = d[(d["country"] == c) & (d["item_id"] == i)]
                k = len(items[i]["options"])
                arr.append([opt_index(o, k) if opt_index(o, k) is not None else -1
                            for o in g["option"]])
            A = np.array(arr).T  # (n_gen, n_items)
            A = A[(A >= 0).all(axis=1)]
            if len(A): gen_prof[c] = A
        res["individuation"][a] = individuation(gen_prof)
        res["exaggeration"][a] = exaggeration(mprof[a], hprof)

    # H4: separability lower for retrieval/forecast than label (descriptive;
    # single value per arm -> report difference vs human baseline)
    res["H4"] = {"label": res["knn_separability"]["label"],
                 "retrieval": res["knn_separability"]["retrieval"],
                 "forecast": res["knn_separability"]["forecast"],
                 "human": res["knn_separability_human"]}

    # --- arm 0: anglocentric default vs Great Britain modal profile
    d0 = df[(df["run_id"].str.startswith("M-")) & (df["arm"] == "neutral")]
    gb_modal = {i: int(np.argmax(items[i]["human_dist"]["Great Britain"]))
                for i in item_ids}
    match = []
    for i in item_ids:
        g = d0[d0["item_id"] == i]
        k = len(items[i]["options"])
        modes = [opt_index(o, k) for o in g["option"]]
        modes = [x for x in modes if x is not None]
        if modes:
            match.append(float(np.mean([x == gb_modal[i] for x in modes])))
    res["arm0_gb_match"] = float(np.mean(match)) if match else None

    json.dump(res, open("outputs/results.json", "w"), indent=1)
    print(json.dumps(res, indent=1)[:3000])

if __name__ == "__main__":
    main()
