"""XGBoost decoding of new / recent / consolidated images from NSD fMRI, set up like the sEEG model.

Mirrors analysis files/XGB-Initiation_TPE_all_trials.py, with Glasser
(HCP-MMP1) parcels instead of channel band powers as features.

Features
  Per trial the single-trial beta (betas_fithrf_GLMdenoise_RR, fsaverage,
  z-scored per vertex within session; --psc: raw percent signal change)
  averaged over the vertices of each HCP-MMP1 parcel (extract_parcel_betas.py).
  --hemispheres mean (default) averages left and right, as the sEEG regions
  pool both hemispheres (180 features); "separate" keeps them (360, L_/R_).
  Columns are named "<parcel> | BOLD1", so the SHAP scripts in shap analysis/
  (region | band+second) read them as band "BOLD", second 1.

Labels (as in the sEEG data, image_category)
  1 new, 2 recent (last shown <= --hours ago), 3 consolidated (longer ago);
  correct responses 1/2/3, errors -1/-2/-3, no response 4/5/6.
  --max-lag-days drops repeats last shown longer ago (13.18 = longest sEEG lag).
  Only correct trials train and test the model; errors and no-response trials
  are predicted and explained afterwards, as for sEEG.

Pipeline (same as sEEG)
  - 50/50 train/test split, stratified on participant x session x category
    (sEEG: subject_stratification + image_category), random_state 42
  - 25 % of the training set held out as a fixed validation set for the
    hyperparameter search; the rest split into fit/eval (eval_fraction is a
    hyperparameter) for early stopping
  - sample weights balanced over participant x category (subject_categories)
  - XGBClassifier(objective="multi:softprob", eval_metric="auc", num_class=3)
  - hyperopt ATPE as in the sEEG script (needs lightgbm; --algo tpe without it), loss =
    -macro one-vs-one AUC on the validation set, searched until --patience
    evaluations bring no improvement (sEEG: hyperparams_iterations = 50)
  - SHAP (TreeExplainer, interventional) on test, error and no-response
    trials, background balanced over participant x category
  - shuffle control: labels (with the stratification and weighting labels)
    rolled within participant by a shift, the whole fit repeated with the best
    hyperparameters; test confusion matrices saved in batches

Hyperparameter space: --space ephys uses the sEEG ranges unchanged; the
default "fmri" keeps the same parameters with ranges that are feasible on
~100k trials (the sEEG ranges allow 60000 trees of depth 35000).

Output (--out-dir, default data/nsd/xgb, git-ignored), <name> = nsd_fmri_xgboost[_options]:
  <name>.pkl                 dict like the sEEG pickle: features (+ subject_stratification
                             as last column), trial_labels, models {xgboost, hyperparameters,
                             train_data, test_data, error_data, unanswered_data,
                             test/error/unanswered_shap_values (trials x features x classes)},
                             results {accuracies, AUCs, confusion matrices}
  <name>_hyperopt.pkl        hyperopt Trials (reused with --continue-search)
  <name>_shuffle_batch_K.pkl permutated_confusion_mat, roll_shift[, shuffled_models]

Examples
  python3 xgb_fmri.py --quick                  # small smoke test (2 participants, few evals)
  python3 xgb_fmri.py --max-lag-days 13.18     # full run in the sEEG lag window
  python3 xgb_fmri.py --shuffles 0             # model + SHAP only
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CATEGORY = {"new": 1, "recent": 2, "consolidated": 3}
GROW_POLICY = ["depthwise", "lossguide"]
INT_PARAMS = ["max_depth", "early_stopping_rounds", "n_estimators", "max_leaves", "max_bin"]


# ----------------------------------------------------------------------------- data

def trial_table(beh: Path, hours: float, max_lag_days: float | None) -> pd.DataFrame:
    """Behaviour in presentation order with trial type and showing number (as plot_recent_vs_consolidated.py)."""
    b = pd.read_csv(beh, sep="\t").sort_values(["SESSION", "RUN", "TRIAL"]).reset_index(drop=True)
    since = b.sort_values("TIME").groupby("73KID")["TIME"].diff() * 24          # hours, NaN on first showing
    b["since_last_h"] = since.reindex(b.index)
    b["type"] = np.where(b.ISOLD == 0, "new", np.where(b.since_last_h > hours, "consolidated", "recent"))
    if max_lag_days is not None:
        b.loc[b.since_last_h > max_lag_days * 24, "type"] = "excluded"
    b["repetition"] = b.sort_values("TIME").groupby("73KID").cumcount().reindex(b.index)
    return b


def load_subject(data_dir: Path, subject: int, args, parcels: list[str]) -> pd.DataFrame | None:
    beh = trial_table(data_dir / f"subj{subject:02d}_responses.tsv", args.hours, args.max_lag_days)
    suffix = "_psc" if args.psc else ""
    parts = []
    for session, trials in beh.groupby("SESSION"):
        files = [data_dir / f"subj{subject:02d}" / f"session{session:02d}_{h}{suffix}.npy" for h in ("lh", "rh")]
        if not all(f.exists() for f in files):
            continue
        lh, rh = (np.load(f) for f in files)
        assert len(lh) == len(trials), (subject, session, len(lh), len(trials))
        if args.hemispheres == "mean":
            x, names = np.nanmean([lh, rh], 0), parcels
        else:
            x, names = np.hstack([lh, rh]), [f"L_{p}" for p in parcels] + [f"R_{p}" for p in parcels]
        df = pd.DataFrame(x.astype(np.float32), columns=[f"{n} | BOLD1" for n in names])
        t = trials.reset_index(drop=True)
        cat = t["type"].map(CATEGORY)
        correct = t["ISCORRECT"]
        # sEEG coding: correct 1/2/3, error -1/-2/-3, no response 4/5/6
        cat = np.where(correct == 1, cat, np.where(correct == 0, -cat, cat + 3))
        df["subject_stratification"] = 1000 * subject + session
        df.index = pd.MultiIndex.from_arrays(
            [t["73KID"].to_numpy(), t["repetition"].to_numpy(), cat, [f"subj{subject:02d}"] * len(t)],
            names=["pic", "repetition", "image_category", "subject"])
        parts.append(df[(t["type"] != "excluded").to_numpy()])
    if not parts:
        return None
    return pd.concat(parts)


def load_all(args) -> pd.DataFrame:
    parcels = (args.data_dir / "parcels.txt").read_text().split()
    frames = []
    for s in args.subjects:
        df = load_subject(args.data_dir, s, args, parcels)
        if df is None:
            print(f"subj{s:02d}: no data")
            continue
        if args.max_trials_per_subject and len(df) > args.max_trials_per_subject:
            keep = np.sort(np.random.default_rng(0).choice(len(df), args.max_trials_per_subject, replace=False))
            df = df.iloc[keep]                                   # presentation order kept (for the roll shuffle)
        c = pd.Series(df.index.get_level_values("image_category")).value_counts()
        print(f"subj{s:02d}: {df.subject_stratification.nunique()} sessions, correct new/recent/consolidated "
              f"{c.get(1, 0)}/{c.get(2, 0)}/{c.get(3, 0)}, errors {c[c.index < 0].sum()}, "
              f"no response {c[c.index > 3].sum()}")
        frames.append(df)
    return pd.concat(frames)


# ----------------------------------------------------------------------------- model helpers

def make_space(kind: str, min_eval_fraction: float) -> dict:
    space = {
        "gamma": hp.uniform("gamma", 0, 15),
        "reg_lambda": hp.uniform("reg_lambda", 0, 2000),
        "reg_alpha": hp.uniform("reg_alpha", 0.001, 50),
        "min_child_weight": hp.uniform("min_child_weight", 0, 200),
        "subsample": hp.uniform("subsample", 0.01, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.01, 1),
        "colsample_bylevel": hp.uniform("colsample_bylevel", 0.01, 1),
        "colsample_bynode": hp.uniform("colsample_bynode", 0.01, 1),
        "eval_fraction": hp.loguniform("eval_fraction", np.log(min_eval_fraction), np.log(0.5)),
        "grow_policy": hp.choice("grow_policy", [0, 1]),
    }
    if kind == "ephys":   # the sEEG ranges unchanged
        space |= {
            "max_depth": hp.quniform("max_depth", 1, 35000, 1),
            "learning_rate": hp.uniform("learning_rate", 0, 6),
            "early_stopping_rounds": hp.quniform("early_stopping_rounds", 100, 35000, 1),
            "n_estimators": hp.quniform("n_estimators", 20, 60000, 1),
            "max_delta_step": hp.uniform("max_delta_step", 0, 150),
            "max_leaves": hp.qloguniform("max_leaves", -5, np.log(1500), 1),
            "max_bin": hp.qloguniform("max_bin", np.log(2), np.log(4000), 1),
        }
    else:                 # same parameters, ranges feasible for ~100k trials
        space |= {
            "max_depth": hp.quniform("max_depth", 1, 12, 1),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.5)),
            "early_stopping_rounds": hp.quniform("early_stopping_rounds", 20, 300, 1),
            "n_estimators": hp.quniform("n_estimators", 50, 4000, 1),
            "max_delta_step": hp.uniform("max_delta_step", 0, 10),
            "max_leaves": hp.qloguniform("max_leaves", np.log(2), np.log(512), 1),
            "max_bin": hp.qloguniform("max_bin", np.log(16), np.log(512), 1),
        }
    return space


def to_params(space: dict) -> tuple[dict, float]:
    """hyperopt point -> XGBClassifier kwargs, eval_fraction."""
    p = dict(space)
    eval_fraction = float(p.pop("eval_fraction"))
    for k in INT_PARAMS:
        p[k] = int(p[k])
    p["max_leaves"] = max(p["max_leaves"], 0)
    if not isinstance(p["grow_policy"], str):
        p["grow_policy"] = GROW_POLICY[int(p["grow_policy"])]
    return p, eval_fraction


def balanced_weights(groups_fit, groups_eval):
    """Balanced class weights over participant x category (as compute_class_weight in the sEEG script)."""
    classes = np.unique(groups_fit)
    w = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=groups_fit)))
    return np.array([w[g] for g in groups_fit]), np.array([w.get(g, 1.0) for g in groups_eval])


def fit_model(params, eval_fraction, X, y, groups, seed, n_jobs):
    fit_X, eval_X, fit_y, eval_y, g_fit, g_eval = train_test_split(
        X, y, groups, test_size=eval_fraction, stratify=y, random_state=seed)
    w_fit, w_eval = balanced_weights(g_fit, g_eval)
    model = xgb.XGBClassifier(**params, num_class=3, objective="multi:softprob", eval_metric="auc",
                              tree_method="hist", n_jobs=n_jobs, random_state=seed)
    model.fit(fit_X, fit_y, verbose=False, sample_weight=w_fit, sample_weight_eval_set=[w_fit, w_eval],
              eval_set=[(fit_X, fit_y), (eval_X, eval_y)])
    return model


def auc(y, proba):
    return roc_auc_score(y, proba, average="macro", multi_class="ovo", labels=[0, 1, 2])


def merge_rare(strata: np.ndarray, fallback: np.ndarray, min_count: int = 2) -> np.ndarray:
    """Strata with fewer than min_count trials fall back to participant x category (sEEG: sessions joined)."""
    strata = strata.astype(object).copy()
    values, counts = np.unique(strata, return_counts=True)
    rare = np.isin(strata, values[counts < min_count])
    strata[rare] = fallback[rare]
    return strata


def shap_values(model, background, X) -> np.ndarray:
    import shap
    if len(X) == 0:
        return np.zeros((0, X.shape[1], 3), np.float32)
    explainer = shap.TreeExplainer(model, data=background) if background is not None else shap.TreeExplainer(model)
    v = explainer.shap_values(X)
    v = np.stack(v, -1) if isinstance(v, list) else np.asarray(v)          # -> trials x features x classes
    return v.astype(np.float32)


def roll_within(values, subjects, shift):
    """Roll values by shift positions separately within each participant (sEEG roll_within_subject)."""
    values = np.asarray(values)
    out = values.copy()
    for s in np.unique(subjects):
        idx = np.flatnonzero(subjects == s)
        out[idx] = np.roll(values[idx], shift % len(idx))
    return out


def shuffle_run(ind, shift, X, category, strata, groups, subjects, params, eval_fraction, test_size, n_jobs,
                keep_model):
    y = roll_within(category, subjects, shift).astype(int) - 1
    s_strata = roll_within(strata, subjects, shift)
    s_groups = roll_within(groups, subjects, shift)
    tr_X, te_X, tr_y, te_y, tr_g, _ = train_test_split(X, y, s_groups, stratify=s_strata, test_size=test_size,
                                                       random_state=ind)
    model = fit_model(params, eval_fraction, tr_X, tr_y, tr_g, ind, n_jobs)
    cm = confusion_matrix(te_y, model.predict(te_X), labels=[0, 1, 2])
    return cm, shift, auc(te_y, model.predict_proba(te_X)), (model if keep_model else None)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "nsd")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data" / "nsd" / "xgb")
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("--hours", type=float, default=3.0, help="recent/consolidated boundary")
    ap.add_argument("--max-lag-days", type=float, help="drop repeats last shown longer ago (13.18 = sEEG range)")
    ap.add_argument("--hemispheres", choices=["mean", "separate"], default="mean")
    ap.add_argument("--psc", action="store_true", help="raw percent signal change instead of session z-scored betas")
    ap.add_argument("--space", choices=["fmri", "ephys"], default="fmri")
    ap.add_argument("--algo", choices=["atpe", "tpe"], default="atpe")
    ap.add_argument("--patience", type=int, default=50,
                    help="stop the search after this many evaluations without improvement (sEEG: 50); 0 = no search")
    ap.add_argument("--max-evals", type=int, default=500, help="upper limit of hyperopt evaluations")
    ap.add_argument("--continue-search", action="store_true", help="continue the saved hyperopt Trials")
    ap.add_argument("--background-per-cell", type=int, default=20,
                    help="SHAP background: training trials per participant x category (0 = tree_path_dependent)")
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--shuffles", type=int, default=1000)
    ap.add_argument("--shuffle-batch", type=int, default=8)
    ap.add_argument("--keep-shuffle-models", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=-1, help="threads per XGBoost fit")
    ap.add_argument("--max-trials-per-subject", type=int, help="random subset per participant (testing)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: participants 2 and 3, 3000 trials each, 6 evals, 16 shuffles")
    args = ap.parse_args()
    if args.quick:
        args.subjects, args.max_trials_per_subject = [2, 3], 3000
        args.patience, args.max_evals, args.shuffles, args.shuffle_batch = 3, 6, 16, 4
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = ("nsd_fmri_xgboost" + (f"_maxlag{args.max_lag_days:g}d" if args.max_lag_days is not None else "")
            + ("_hemis" if args.hemispheres == "separate" else "") + ("_psc" if args.psc else "")
            + ("_ephysspace" if args.space == "ephys" else "") + ("_quick" if args.quick else ""))
    out = args.out_dir / name

    # ------------------------------------------------------------------ data and splits
    overall_df = load_all(args)
    image_labels = overall_df.index.get_level_values("image_category").to_numpy()
    features = [c for c in overall_df.columns if c != "subject_stratification"]
    correct_df = overall_df[(image_labels > 0) & (image_labels < 4)]
    category = correct_df.index.get_level_values("image_category").to_numpy().astype(int)
    subjects = correct_df.index.get_level_values("subject").to_numpy()
    groups = np.array([f"{s}_{c}" for s, c in zip(subjects, category)])                     # participant x category
    strata = merge_rare(np.array([f"{s}_{g}" for s, g in zip(correct_df.subject_stratification, category)]), groups)
    X = correct_df[features]
    print(f"{len(X)} correct trials, {len(features)} features, {len(np.unique(subjects))} participants")

    test_size, val_size = 0.5, 0.25
    train_X, test_X, train_y, test_y, train_g, test_g = train_test_split(
        X, category - 1, groups, stratify=strata, test_size=test_size, random_state=42)
    sub_X, val_X, sub_y, val_y, sub_g, _ = train_test_split(
        train_X, train_y, train_g, stratify=train_y, test_size=val_size, random_state=42)
    min_eval_fraction = 1 / (np.bincount(train_y).min() * (1 - val_size))

    # ------------------------------------------------------------------ hyperparameter search
    def objective(space):
        params, eval_fraction = to_params(space)
        try:
            model = fit_model(params, eval_fraction, sub_X, sub_y, sub_g, 0, args.n_jobs)
            return {"loss": -auc(val_y, model.predict_proba(val_X)), "status": STATUS_OK}
        except Exception as e:                       # e.g. a fit that fails for extreme parameters
            print("failed evaluation:", e)
            return {"loss": 0.0, "status": STATUS_OK}

    hyper_path = Path(f"{out}_hyperopt.pkl")
    if args.continue_search and hyper_path.exists():
        trials = pickle.load(open(hyper_path, "rb"))
    else:
        trials = Trials()
    if args.algo == "atpe":
        from hyperopt import atpe
        algo = atpe.suggest
    else:
        algo = tpe.suggest
    space = make_space(args.space, min_eval_fraction)
    if args.patience:
        best_loss, since_best = (trials.best_trial["result"]["loss"] if len(trials) else np.inf), 0
        while since_best < args.patience and len(trials) < args.max_evals:
            t0 = time.time()
            fmin(objective, space, algo=algo, max_evals=len(trials) + 1, trials=trials,
                 rstate=np.random.default_rng(len(trials)), show_progressbar=False)
            loss = trials.trials[-1]["result"]["loss"]
            since_best = 0 if loss < best_loss else since_best + 1
            best_loss = min(best_loss, loss)
            print(f"eval {len(trials):3d}: val AUC {-loss:.4f} (best {-best_loss:.4f}, {time.time() - t0:.0f} s)")
            pickle.dump(trials, open(hyper_path, "wb"))
        best = {k: v[0] for k, v in trials.best_trial["misc"]["vals"].items()}
        best = {k: (GROW_POLICY[int(v)] if k == "grow_policy" else v) for k, v in best.items()}
    elif hyper_path.exists():
        trials = pickle.load(open(hyper_path, "rb"))
        best = {k: v[0] for k, v in trials.best_trial["misc"]["vals"].items()}
    else:
        raise SystemExit("--patience 0 needs a saved hyperopt file")
    params, eval_fraction = to_params(best)
    print("best hyperparameters:", params, "eval_fraction", round(eval_fraction, 4))

    # ------------------------------------------------------------------ final model
    model = fit_model(params, eval_fraction, train_X, train_y, train_g, 42, args.n_jobs)
    error_df = overall_df[image_labels < 0][features]
    error_y = (-error_df.index.get_level_values("image_category").to_numpy()).astype(int) - 1
    unanswered_df = overall_df[image_labels > 3][features]
    unanswered_y = unanswered_df.index.get_level_values("image_category").to_numpy().astype(int) - 4
    results = {}
    for label, x, y in (("train", train_X, train_y), ("test", test_X, test_y), ("error", error_df, error_y),
                        ("unanswered", unanswered_df, unanswered_y)):
        if len(x) == 0:
            continue
        proba = model.predict_proba(x)
        pred = proba.argmax(1)
        results[label] = {"accuracy": accuracy_score(y, pred),
                          "balanced_accuracy": np.mean([np.mean(pred[y == k] == k) for k in np.unique(y)]),
                          "auc": auc(y, proba) if len(np.unique(y)) == 3 else np.nan,
                          "confusion": confusion_matrix(y, pred, labels=[0, 1, 2])}
        print(f"{label:10s} n={len(y):6d}  accuracy {results[label]['accuracy']:.3f}  balanced "
              f"{results[label]['balanced_accuracy']:.3f}  AUC {results[label]['auc']:.3f}")
    print("features never used:", int((model.feature_importances_ == 0).sum()), "of", len(features))

    summary = {"features": overall_df, "trial_labels": image_labels,
               "models": {"xgboost": model, "hyperparameters": best, "train_data": train_X.index,
                          "test_data": test_X.index, "error_data": error_df.index,
                          "unanswered_data": unanswered_df.index},
               "results": results, "settings": vars(args) | {"data_dir": str(args.data_dir),
                                                            "out_dir": str(args.out_dir)}}
    save = lambda: pickle.dump(summary, open(f"{out}.pkl", "wb"))
    save()

    # ------------------------------------------------------------------ SHAP
    if not args.no_shap:
        if args.background_per_cell:
            cells = pd.Series(train_g, index=range(len(train_g)))
            idx = np.concatenate([np.random.default_rng(42).choice(
                ix, args.background_per_cell, replace=len(ix) < args.background_per_cell)
                for _, ix in cells.groupby(cells).groups.items()])
            background = train_X.iloc[idx]
        else:
            background = None
        for label, x in (("test", test_X), ("error", error_df), ("unanswered", unanswered_df)):
            t0 = time.time()
            summary["models"][f"{label}_shap_values"] = shap_values(model, background, x)
            print(f"SHAP {label}: {len(x)} trials, {time.time() - t0:.0f} s")
            save()

    # ------------------------------------------------------------------ shuffle control
    if args.shuffles:
        n_min = int(pd.Series(subjects).value_counts().min())
        shifts = np.resize(np.random.default_rng(0).permutation(np.arange(1, n_min)), args.shuffles)
        batches = int(np.ceil(args.shuffles / args.shuffle_batch))
        threads = max(1, (args.n_jobs if args.n_jobs > 0 else 4) // args.shuffle_batch)
        for b in range(batches):
            path = Path(f"{out}_shuffle_batch_{b}.pkl")
            if path.exists():
                continue
            inds = range(b * args.shuffle_batch + 1, min((b + 1) * args.shuffle_batch, args.shuffles) + 1)
            res = Parallel(n_jobs=args.shuffle_batch)(delayed(shuffle_run)(
                i, int(shifts[i - 1]), X, category, strata, groups, subjects, params, eval_fraction, test_size,
                threads, args.keep_shuffle_models) for i in inds)
            pickle.dump({"shuffle_batch": b, "permutated_confusion_mat": [r[0] for r in res],
                         "roll_shift": [r[1] for r in res], "auc": [r[2] for r in res],
                         "shuffled_models": [r[3] for r in res]}, open(path, "wb"))
            print(f"shuffle batch {b + 1}/{batches}: accuracies "
                  + ", ".join(f"{np.trace(r[0]) / r[0].sum():.3f}" for r in res))
        cms = [cm for b in range(batches) for cm in pickle.load(open(f"{out}_shuffle_batch_{b}.pkl", "rb"))
               ["permutated_confusion_mat"]]
        aucs = np.array([a for b in range(batches) for a in pickle.load(open(f"{out}_shuffle_batch_{b}.pkl", "rb"))
                         ["auc"]])
        acc = np.array([np.trace(c) / c.sum() for c in cms])
        bal = np.array([np.mean(np.diag(c) / c.sum(1)) for c in cms])
        summary["results"]["shuffle"] = {"accuracies": acc, "balanced_accuracies": bal, "aucs": aucs}
        for key, null in (("accuracy", acc), ("balanced_accuracy", bal), ("auc", aucs)):
            real = results["test"][key]
            p = (1 + np.sum(null >= real)) / (1 + len(null))
            summary["results"]["shuffle"][f"p_{key}"] = p
            print(f"test {key} {real:.3f}; shuffled 95th percentile {np.percentile(null, 95):.3f}; "
                  f"p = {p:.4f} ({len(null)} shuffles)")
        save()
    print("wrote", f"{out}.pkl")


if __name__ == "__main__":
    main()
