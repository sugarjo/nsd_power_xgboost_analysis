"""Download NSD single-trial betas and reduce them to HCP-MMP1 parcel averages.

For every subject, scan session and hemisphere this script
  1. downloads the fsaverage betas (betas_fithrf_GLMdenoise_RR, percent signal
     change, 163842 vertices x 750 trials) from the public NSD bucket,
  2. z-scores every vertex across the 750 trials of the session (the usual
     NSD practice; removes session-to-session differences in signal level),
  3. averages the z-scored betas over the vertices of each HCP-MMP1 parcel,
  4. saves a 750 x 180 array (trials in presentation order, parcels in
     annotation order) and deletes the download.
Sessions that are already saved are skipped, so the script can be restarted.

Written to --out-dir:
  subjXX/sessionYY_{lh,rh}.npy   float32, trials x parcels
  parcels.txt                     parcel names, one per line (column order)
"""

import argparse
import os
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer as fs
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BUCKET = "https://natural-scenes-dataset.s3.amazonaws.com"
ANNOT = "https://raw.githubusercontent.com/poldrack/GOBS/master/extract/HCP-MMP1/{hemi}.HCP-MMP1.annot"


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def parcel_index(cache: Path, hemi: str):
    """Vertex -> parcel column (-1 for medial wall / unknown) and the parcel names."""
    labels, _, names = fs.read_annot(fetch(ANNOT.format(hemi=hemi), cache / f"{hemi}.HCP-MMP1.annot"))
    names = [n.decode() for n in names]
    keep = [i for i, n in enumerate(names) if n != "???"]
    column = np.full(len(names) + 1, -1)
    column[keep] = np.arange(len(keep))
    parcels = [names[i][2:-4] for i in keep]          # "L_V1_ROI" -> "V1"
    return column[labels], parcels


def session_parcels(subject: int, session: int, hemi: str, vertex_parcel, n_parcels, tmp_dir: Path):
    url = (f"{BUCKET}/nsddata_betas/ppdata/subj{subject:02d}/fsaverage/betas_fithrf_GLMdenoise_RR/"
           f"{hemi}.betas_session{session:02d}.mgh")
    path = tmp_dir / f"s{subject}_{session}_{hemi}.mgh"
    try:
        fetch(url, path)
        betas = np.asarray(nib.load(path).dataobj, dtype=np.float32).reshape(len(vertex_parcel), -1)
    finally:
        if path.exists():
            path.unlink()
    mean = betas.mean(1, keepdims=True)
    std = betas.std(1, keepdims=True)
    std[std == 0] = np.nan                                 # flat vertices (medial wall) -> NaN
    z = (betas - mean) / std
    out = np.full((z.shape[1], n_parcels), np.nan, np.float32)
    valid = (vertex_parcel >= 0) & np.isfinite(std[:, 0])
    for p in range(n_parcels):
        sel = valid & (vertex_parcel == p)
        if sel.any():
            out[:, p] = z[sel].mean(0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("--out-dir", type=Path, default=REPO / "data" / "nsd")
    ap.add_argument("--workers", type=int, default=3, help="sessions processed in parallel (~2 GB RAM each)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    index = {h: parcel_index(args.out_dir, h) for h in ("lh", "rh")}
    assert index["lh"][1] == index["rh"][1]
    (args.out_dir / "parcels.txt").write_text("\n".join(index["lh"][1]) + "\n")

    jobs = []
    for subject in args.subjects:
        behav = fetch(f"{BUCKET}/nsddata/ppdata/subj{subject:02d}/behav/responses.tsv",
                      args.out_dir / f"subj{subject:02d}_responses.tsv")
        sessions = sorted(pd.read_csv(behav, sep="\t", usecols=["SESSION"]).SESSION.unique())
        (args.out_dir / f"subj{subject:02d}").mkdir(exist_ok=True)
        for session in sessions:
            for hemi in ("lh", "rh"):
                out = args.out_dir / f"subj{subject:02d}" / f"session{session:02d}_{hemi}.npy"
                if not out.exists():
                    jobs.append((subject, session, hemi, out))
    print(f"{len(jobs)} session files to process")

    tmp_dir = Path(tempfile.mkdtemp(dir=args.out_dir))

    def run(job):
        subject, session, hemi, out = job
        column, parcels = index[hemi]
        arr = session_parcels(subject, session, hemi, column, len(parcels), tmp_dir)
        np.save(out.with_suffix(".tmp.npy"), arr)
        os.replace(out.with_suffix(".tmp.npy"), out)
        return job

    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            try:
                s, se, h, _ = f.result()
                print(f"[{i}/{len(jobs)}] subj{s:02d} session{se:02d} {h}", flush=True)
            except Exception as e:                        # keep going; a rerun retries what failed
                print(f"[{i}/{len(jobs)}] failed: {e!r}", flush=True)


if __name__ == "__main__":
    main()
