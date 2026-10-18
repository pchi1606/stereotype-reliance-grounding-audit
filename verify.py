"""Verifier: checks recomputed results against expected headline values.

Compares outputs/results.json (freshly computed by code/analyze.py)
against expected/values.json within tolerance. Prints PASS/FAIL per check.
"""
import json
import sys

def main():
    res = json.load(open("outputs/results.json"))
    exp = json.load(open("expected/values.json"))
    tol = exp.get("tolerance", 0.01)
    checks = []
    # Hypothesis p-values
    h2 = {h["name"]: h for h in res["H2"]}
    h3 = {h["name"]: h for h in res["H3"]}
    checks.append(("H2a p", h2["H2a label-vs-retrieval"]["p"], exp["H2a_p"]))
    checks.append(("H2b p", h2["H2b label-vs-forecast"]["p"], exp["H2b_p"]))
    checks.append(("H3a p", h3["H3a retrieval-vs-label"]["p"], exp["H3a_p"]))
    checks.append(("H3b p", h3["H3b forecast-vs-label"]["p"], exp["H3b_p"]))
    # Means
    for arm in ["label", "retrieval", "forecast"]:
        checks.append((f"Wasserstein {arm}",
                       res["mean_wasserstein"][arm],
                       exp["mean_wasserstein"][arm]))
        checks.append((f"Entropy ratio {arm}",
                       res["mean_entropy_ratio"][arm],
                       exp["mean_entropy_ratio"][arm]))
        checks.append((f"kNN {arm}",
                       res["knn_separability"][arm],
                       exp["knn_separability"][arm]))
    checks.append(("n_cells", res["n_cells"], exp["n_cells"]))
    ok = True
    for name, got, want in checks:
        good = abs(got - want) <= tol
        ok = ok and good
        print(f"{'PASS' if good else 'FAIL'} {name}: got {got:.4f}, want {want:.4f}")
    print("OVERALL:", "PASS" if ok else "FAIL")
    return ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
