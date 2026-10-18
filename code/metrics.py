"""Preregistered metrics (§4 of PREREGISTRATION.md).

Conventions:
  H[c][i] = human distribution (list over K options) for country c, item i
  M[a][c][i] = model empirical distribution from 20 generations, arm a
"""
import numpy as np
from scipy.stats import wasserstein_distance, entropy

def as_dist(counts, k):
    counts = np.asarray(counts, dtype=float)
    t = counts.sum()
    return counts / t if t > 0 else np.full(k, 1.0 / k)

def wasserstein_ordinal(p, q):
    """1-Wasserstein on option positions 1..K."""
    k = len(p)
    return wasserstein_distance(np.arange(1, k + 1), np.arange(1, k + 1),
                                u_weights=p, v_weights=q)

def tvd(p, q):
    return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum()

def entropy_ratio(p_model, p_human):
    hm = entropy(p_human) / np.log(len(p_human))
    mm = entropy(p_model) / np.log(len(p_model))
    return mm / hm if hm > 1e-9 else (1.0 if mm < 1e-9 else np.inf)

def country_profile_modal(counts_by_item):
    """Profile vector = modal option index per item (ties -> lowest index)."""
    return np.array([int(np.argmax(c)) for c in counts_by_item])

def knn_loo_accuracy(profiles):
    """1-NN leave-one-out accuracy predicting country from modal profile.

    profiles: dict country -> np.array (n_items,). Returns accuracy in [0,1].
    """
    countries = list(profiles.keys())
    X = np.stack([profiles[c] for c in countries])
    # hamming distance
    D = (X[:, None, :] != X[None, :, :]).mean(axis=2)
    np.fill_diagonal(D, np.inf)
    pred = D.argmin(axis=1)
    return float(np.mean([countries[p] == c for p, c in zip(pred, countries)]))

def individuation(generation_profiles):
    """within-country variance / total variance of per-generation profiles.

    generation_profiles: dict country -> (n_gen, n_items) int array.
    Low = caricature (everyone from a country answers identically).
    """
    all_p = np.concatenate(list(generation_profiles.values()), axis=0).astype(float)
    total_var = all_p.var(axis=0).sum()
    if total_var < 1e-12:
        return 0.0
    within = np.mean([p.astype(float).var(axis=0).sum()
                      for p in generation_profiles.values()])
    return float(within / total_var)

def exaggeration(model_profiles, human_profiles):
    """mean_c dist(model_c, human_c) / mean pairwise dist(human countries).

    Profiles are modal vectors; distance = hamming distance.
    >1 = model pushes the country further out than real countries differ.
    """
    common = [c for c in model_profiles if c in human_profiles]
    num = np.mean([(model_profiles[c] != human_profiles[c]).mean() for c in common])
    hs = [human_profiles[c] for c in common]
    den = np.mean([(a != b).mean() for x, a in enumerate(hs)
                   for b in hs[x + 1:]]) if len(hs) > 1 else np.nan
    return float(num / den) if den > 0 else np.nan

def knn_loo_accuracy_matrix(X, y):
    """1-NN leave-one-out accuracy over per-generation profiles.

    X: (n_gen, n_items) int array of option indices; y: (n_gen,) country
    labels. Hamming distance; ties -> lowest index (deterministic).
    Unlike the modal-profile variant, this is non-degenerate when each
    country has many generations (chance = 1 / n_countries).
    """
    X = np.asarray(X); y = np.asarray(y)
    D = (X[:, None, :] != X[None, :, :]).mean(axis=2)
    np.fill_diagonal(D, np.inf)
    pred = D.argmin(axis=1)
    return float(np.mean(y[pred] == y))
