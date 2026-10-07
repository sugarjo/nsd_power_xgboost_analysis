"""Subject-balanced bagged TabPFN: train on balanced bags, test on a held-out validation set.

TabPFN's fit() takes no sample weights; it uses the training rows as context.
Each subject is therefore given equal weight by giving it the same number of
rows in every bag: for every (subject, class) cell of the training set, a bag
draws `--n-per-cell` trials. Cells with more trials are subsampled without
replacement (a new subset per bag); cells with fewer are filled up with
  duplicate  random repeats of the cell's own trials, or
  interp     synthetic trials a + u * (b - a), u ~ U(0, 1), between two
             trials of the same subject and class; an entry is NaN when it is
             NaN in either source trial, so no values are invented for regions
             the subject had no channel in, or
  none       no filling (small cells keep all their trials and stay smaller).
A TabPFN model is fitted on each bag and the validation probabilities are
averaged over the bags. Only training rows are ever resampled.

Input: the pickle's "features" DataFrame, whose index has the levels
"subject" and "image_category" (see behavioral matrices/plot_response_matrices.py
for the codes). The class is the image type, |image_category| - 1, i.e.
new (0), recent (1), old (2), the class order of the XGBClassifier. The column
subject_stratification is never used as a feature.

Validation split (--split):
  existing  the XGBoost test set, data["models"]["test_data"] (comparable to
            the XGBoost results); falls back to "within" if it is missing
  within    a random --val-size share of every (subject, class) cell
  subjects  whole subjects held out (--val-size share of the subjects)

Written to --out-dir:
  bagged_tabpfn_predictions.pkl  DataFrame, index = validation trials,
                                 columns p_new, p_recent, p_old
  bagged_tabpfn_per_subject.csv  per-subject validation metrics
  bagged_tabpfn_per_bag.csv      validation score of the running bag average
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

CLASSES = ["new", "recent", "old"]
STRAT_COLUMN = "subject_stratification"


def load(pkl: Path, categories) -> tuple[dict, pd.DataFrame, pd.Series]:
    """Features (trials x numeric features) and class labels for the chosen categories."""
    data = pd.read_pickle(pkl)
    feats = data["features"]
    cat = feats.index.get_level_values("image_category")
    feats = feats[np.isin(cat, categories)]
    X = feats.drop(columns=[STRAT_COLUMN], errors="ignore").select_dtypes("number")
    y = pd.Series(np.abs(X.index.get_level_values("image_category")) - 1, index=X.index, name="label")
    return data, X, y


def split(data: dict, X: pd.DataFrame, y: pd.Series, how: str, val_size: float,
          seed: int) -> tuple[np.ndarray, str]:
    """Boolean mask over the rows of X, True for validation trials, and the split used."""
    subj = X.index.get_level_values("subject").to_numpy()
    if how == "existing":
        test = data.get("models", {}).get("test_data")
        if test is not None:
            is_val = X.index.isin(test)
            if is_val.any():
                return is_val, how
        print("no models/test_data in the pickle; using --split within")
        how = "within"
    rows = np.arange(len(X))
    if how == "within":
        strata = pd.Series(subj).astype(str) + "_" + y.astype(str).to_numpy()
        _, val = train_test_split(rows, test_size=val_size, stratify=strata, random_state=seed)
    else:
        _, val = next(GroupShuffleSplit(1, test_size=val_size, random_state=seed).split(rows, groups=subj))
    is_val = np.zeros(len(X), bool)
    is_val[val] = True
    return is_val, how


def cells(y: np.ndarray, subj: np.ndarray) -> list[np.ndarray]:
    """Row indices of every non-empty (subject, class) cell."""
    out = []
    for s in np.unique(subj):
        for c in np.unique(y[subj == s]):
            out.append(np.flatnonzero((subj == s) & (y == c)))
    return out


def make_bag(X: np.ndarray, y: np.ndarray, cell_rows: list[np.ndarray], n: int,
             upsample: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One subject-balanced bag: n trials from every (subject, class) cell."""
    Xs, ys = [], []
    for r in cell_rows:
        take = rng.choice(r, min(n, len(r)), replace=False)
        parts = [X[take]]
        need = n - len(take)
        if need > 0 and (upsample == "duplicate" or (upsample == "interp" and len(r) < 2)):
            parts.append(X[rng.choice(r, need)])
        elif need > 0 and upsample == "interp":
            a, b = X[rng.choice(r, need)], X[rng.choice(r, need)]
            parts.append(a + rng.uniform(0, 1, (need, 1)).astype(X.dtype) * (b - a))
        cell = np.vstack(parts)
        Xs.append(cell)
        ys.append(np.full(len(cell), y[r[0]]))
    return np.vstack(Xs), np.concatenate(ys)


def make_model(name: str, seed: int, args):
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(n_estimators=args.n_estimators, device=args.device,
                                ignore_pretraining_limits=args.ignore_limits, random_state=seed)
    # Stand-in that also accepts NaN, for testing the pipeline without TabPFN weights.
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=100, random_state=seed)


def predict_proba(model, X: np.ndarray, batch: int) -> np.ndarray:
    """Class probabilities in batches (TabPFN's memory grows with the test size)."""
    proba = np.vstack([model.predict_proba(X[i:i + batch]) for i in range(0, len(X), batch)])
    full = np.zeros((len(X), len(CLASSES)))                 # in case a class was absent
    full[:, model.classes_.astype(int)] = proba
    return full


def scores(y: np.ndarray, proba: np.ndarray, subj: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Per-subject metrics, and summaries where every subject weighs the same."""
    rows = []
    for s in np.unique(subj):
        m = subj == s
        row = {"subject": s, "n_trials": int(m.sum()),
               "balanced_accuracy": balanced_accuracy_score(y[m], proba[m].argmax(1))}
        if len(np.unique(y[m])) == len(CLASSES):
            row["auc_ovr_macro"] = roc_auc_score(y[m], proba[m], multi_class="ovr")
        row["log_loss"] = log_loss(y[m], proba[m], labels=range(len(CLASSES)))
        rows.append(row)
    per_subject = pd.DataFrame(rows).set_index("subject")
    w = 1.0 / pd.Series(subj).map(pd.Series(subj).value_counts()).to_numpy()
    summary = {
        "mean_subject_balanced_accuracy": per_subject["balanced_accuracy"].mean(),
        "mean_subject_auc": per_subject.get("auc_ovr_macro", pd.Series(dtype=float)).mean(),
        "pooled_auc_subject_weighted": roc_auc_score(y, proba, multi_class="ovr", sample_weight=w),
        "pooled_log_loss_subject_weighted": log_loss(y, proba, sample_weight=w, labels=range(len(CLASSES))),
    }
    return per_subject, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "real_data_bipolar_xgboost_3w.pkl")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    ap.add_argument("--categories", type=int, nargs="+", default=[1, 2, 3],
                    help="image_category codes to use (default: correct trials 1 2 3)")
    ap.add_argument("--split", choices=["existing", "within", "subjects"], default="existing")
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--n-bags", type=int, default=10)
    ap.add_argument("--n-per-cell", type=int, default=None,
                    help="trials per (subject, class) cell in a bag (default: median cell size, "
                         "capped so a bag has at most --max-context rows)")
    ap.add_argument("--max-context", type=int, default=10_000)
    ap.add_argument("--upsample", choices=["interp", "duplicate", "none"], default="interp")
    ap.add_argument("--model", choices=["tabpfn", "hgb"], default="tabpfn",
                    help="hgb = sklearn HistGradientBoosting stand-in, for testing the pipeline")
    ap.add_argument("--n-estimators", type=int, default=4, help="TabPFN's own ensemble size per bag")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ignore-limits", action="store_true",
                    help="allow more rows/features than TabPFN was pretrained for")
    ap.add_argument("--batch", type=int, default=2000, help="validation rows per predict call")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data, X, y = load(args.pkl, args.categories)
    is_val, how = split(data, X, y, args.split, args.val_size, args.seed)
    subj = X.index.get_level_values("subject").to_numpy()
    Xa, ya = X.to_numpy(np.float32), y.to_numpy()
    X_tr, y_tr, s_tr = Xa[~is_val], ya[~is_val], subj[~is_val]
    X_va, y_va, s_va = Xa[is_val], ya[is_val], subj[is_val]
    observed = ~np.isnan(X_tr).all(axis=0)          # drop features no training trial has
    X_tr, X_va = X_tr[:, observed], X_va[:, observed]

    cell_rows = cells(y_tr, s_tr)
    sizes = np.array([len(r) for r in cell_rows])
    n = args.n_per_cell or int(min(np.median(sizes), args.max_context // len(cell_rows)))
    print(f"{observed.sum()} features ({(~observed).sum()} all-NaN in training dropped); train {len(X_tr)} trials / {len(np.unique(s_tr))} subjects, "
          f"validation {len(X_va)} trials / {len(np.unique(s_va))} subjects ({how})")
    print(f"{len(cell_rows)} (subject, class) cells of {sizes.min()}-{sizes.max()} trials "
          f"(median {int(np.median(sizes))}); {n} per cell -> {n * len(cell_rows)} rows per bag; "
          f"{(sizes < n).sum()} cells smaller than that (upsample: {args.upsample})")

    rng = np.random.default_rng(args.seed)
    proba_sum = np.zeros((len(X_va), len(CLASSES)))
    per_bag = []
    for b in range(args.n_bags):
        Xb, yb = make_bag(X_tr, y_tr, cell_rows, n, args.upsample, rng)
        model = make_model(args.model, args.seed + b, args)
        model.fit(Xb, yb)
        proba_sum += predict_proba(model, X_va, args.batch)
        _, summary = scores(y_va, proba_sum / (b + 1), s_va)
        per_bag.append({"bags": b + 1, **summary})
        print(f"bag {b + 1}/{args.n_bags}: mean subject balanced accuracy "
              f"{summary['mean_subject_balanced_accuracy']:.3f}, mean subject AUC {summary['mean_subject_auc']:.3f}")

    proba = proba_sum / args.n_bags
    per_subject, summary = scores(y_va, proba, s_va)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(proba, index=X.index[is_val], columns=[f"p_{c}" for c in CLASSES]).to_pickle(
        args.out_dir / "bagged_tabpfn_predictions.pkl")
    per_subject.round(4).to_csv(args.out_dir / "bagged_tabpfn_per_subject.csv")
    pd.DataFrame(per_bag).round(4).to_csv(args.out_dir / "bagged_tabpfn_per_bag.csv", index=False)

    print("\nper subject (validation):")
    print(per_subject.round(3).to_string())
    print("\nsummary (every subject weighs the same):")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
