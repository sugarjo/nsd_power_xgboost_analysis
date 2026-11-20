"""Per-vertex trial-type sums of raw NSD betas, for the Allen et al. (2022) memory analysis.

The paper (cvnlab/nsddatapaper, mainfigures/SCIENCE.MEMORY/SCIENCEMEMORYnotes.m)
uses the fsaverage betas_fithrf_GLMdenoise_RR in percent signal change without
any normalisation, and per vertex a Welch t between hits and correct
rejections, for single sessions and pooled across sessions. A Welch t only
needs n, sum and sum of squares per group, so for every subject, session and
hemisphere this script downloads the betas and stores, per vertex and per group
of correct trials,
  cr            correct rejections (new image, answered new)
  recent        hits, image last shown <= --hours ago
  consolidated  hits, image last shown >  --hours ago
  perm_a/perm_b hits and correct rejections with labels shuffled (sizes kept)
  perm_c/perm_d recent and consolidated hits with labels shuffled
(hits = recent + consolidated). It also stores the raw (unnormalised) parcel
means, trials x parcels, for parcel-level analyses.

Written to --out-dir/subjXX/:
  session<YY>_<hemi>_stats.npz    n (groups), sum, sumsq (groups x vertices, float32)
  session<YY>_<hemi>_psc.npy      trials x parcels, raw percent signal change
Resumable: sessions already saved are skipped.
"""

import argparse
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from extract_parcel_betas import BUCKET, fetch, parcel_index  # noqa: E402

GROUPS = ["cr", "recent", "consolidated", "perm_a", "perm_b", "perm_c", "perm_d"]


def trial_types(beh: Path, hours: float) -> pd.DataFrame:
    """Behaviour in presentation order with cr / recent / consolidated / '' (incorrect)."""
    b = pd.read_csv(beh, sep="\t").sort_values(["SESSION", "RUN", "TRIAL"]).reset_index(drop=True)
    since = b.sort_values("TIME").groupby("73KID")["TIME"].diff() * 24
    b["since_last_h"] = since.reindex(b.index)
    kind = np.where(b.ISOLD == 0, "cr", np.where(b.since_last_h > hours, "consolidated", "recent"))
    b["group"] = np.where(b.ISCORRECT == 1, kind, "")
    return b


def group_masks(groups: np.ndarray, rng) -> dict:
    hit = np.isin(groups, ["recent", "consolidated"])
    masks = {g: groups == g for g in GROUPS[:3]}
    pool = np.flatnonzero(hit | (groups == "cr"))           # shuffled hits vs correct rejections
    shuffled = rng.permutation(pool)
    masks["perm_a"] = np.isin(np.arange(len(groups)), shuffled[:hit.sum()])
    masks["perm_b"] = np.isin(np.arange(len(groups)), shuffled[hit.sum():])
    pool = np.flatnonzero(hit)                               # shuffled recent vs consolidated
    shuffled = rng.permutation(pool)
    n_rec = masks["recent"].sum()
    masks["perm_c"] = np.isin(np.arange(len(groups)), shuffled[:n_rec])
    masks["perm_d"] = np.isin(np.arange(len(groups)), shuffled[n_rec:])
    return masks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("--sessions", type=int, nargs="+", help="only these sessions (default: all)")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data" / "nsd")
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    index = {h: parcel_index(args.out_dir, h) for h in ("lh", "rh")}
    jobs = []
    for subject in args.subjects:
        beh = trial_types(fetch(f"{BUCKET}/nsddata/ppdata/subj{subject:02d}/behav/responses.tsv",
                                args.out_dir / f"subj{subject:02d}_responses.tsv"), args.hours)
        (args.out_dir / f"subj{subject:02d}").mkdir(exist_ok=True)
        for session, trials in beh.groupby("SESSION"):
            if args.sessions and session not in args.sessions:
                continue
            for hemi in ("lh", "rh"):
                out = args.out_dir / f"subj{subject:02d}" / f"session{session:02d}_{hemi}_stats.npz"
                if not out.exists():
                    jobs.append((subject, session, hemi, trials["group"].to_numpy(), out))
    print(f"{len(jobs)} session files to process", flush=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=args.out_dir))

    def run(job):
        subject, session, hemi, groups, out = job
        column, parcels = index[hemi]
        url = (f"{BUCKET}/nsddata_betas/ppdata/subj{subject:02d}/fsaverage/betas_fithrf_GLMdenoise_RR/"
               f"{hemi}.betas_session{session:02d}.mgh")
        path = tmp_dir / f"s{subject}_{session}_{hemi}.mgh"
        try:
            fetch(url, path)
            betas = np.asarray(nib.load(path).dataobj, dtype=np.float32).reshape(len(column), -1)
        finally:
            for p in (path, path.with_suffix(".mgh.part")):
                if p.exists():
                    p.unlink()
        assert betas.shape[1] == len(groups), (subject, session, betas.shape)
        rng = np.random.default_rng([args.seed, subject, session, hemi == "rh"])
        masks = group_masks(groups, rng)
        n = np.array([masks[g].sum() for g in GROUPS])
        b64 = betas.astype(np.float64)
        s = np.stack([b64[:, masks[g]].sum(1) for g in GROUPS]).astype(np.float32)
        ss = np.stack([(b64[:, masks[g]] ** 2).sum(1) for g in GROUPS]).astype(np.float32)
        psc = np.full((betas.shape[1], len(parcels)), np.nan, np.float32)
        for p in range(len(parcels)):
            sel = column == p
            if sel.any():
                psc[:, p] = betas[sel].mean(0)
        psc_out = out.with_name(out.name.replace("_stats.npz", "_psc.npy"))
        np.save(psc_out.with_suffix(".tmp.npy"), psc)
        os.replace(psc_out.with_suffix(".tmp.npy"), psc_out)
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, n=n, sum=s, sumsq=ss, groups=np.array(GROUPS))
        os.replace(tmp, out)
        return job

    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            try:
                s, se, h, _, _ = f.result()
                print(f"[{i}/{len(jobs)}] subj{s:02d} session{se:02d} {h}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(jobs)}] failed: {e!r}", flush=True)


if __name__ == "__main__":
    main()
