"""Load every JSON file in a folder into one pandas DataFrame.

Each file may hold
  - one record (a dict)                       -> one row
  - a list of records (list of dicts)         -> one row per record
  - a dict wrapping such a list, e.g. {"subject": "S01", "trials": [...]}
    -> pass --record-path trials (the other top-level keys are copied onto
       every row; use --meta to choose which ones)

Nested dicts are flattened into columns joined with "." (e.g. "power.alpha").
A "source_file" column records which file each row came from.

Usage:
  python load_jsons.py path/to/folder                     # print a summary
  python load_jsons.py path/to/folder --out data.csv      # also save it
  python load_jsons.py path/to/folder --recursive --record-path trials

or from Python:
  from load_jsons import load_json_folder
  df = load_json_folder("path/to/folder")
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_json_file(path: Path, record_path=None, meta=None) -> pd.DataFrame:
    """One JSON file -> DataFrame (nested dicts flattened)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if record_path is not None:
        if meta is None and isinstance(data, dict):
            # keep every scalar top-level key as metadata on each row
            meta = [k for k, v in data.items()
                    if k != record_path and not isinstance(v, (dict, list))]
        return pd.json_normalize(data, record_path=record_path, meta=meta, errors="ignore")
    if isinstance(data, list) and data and not isinstance(data[0], dict):
        return pd.DataFrame({"value": data})
    return pd.json_normalize(data)


def load_json_folder(folder, pattern="*.json", recursive=False,
                     record_path=None, meta=None) -> pd.DataFrame:
    """All JSON files in `folder` -> one DataFrame, with a "source_file" column."""
    folder = Path(folder)
    files = sorted(folder.rglob(pattern) if recursive else folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} in {folder}")
    frames = []
    for path in files:
        try:
            df = load_json_file(path, record_path, meta)
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as e:
            print(f"skipping {path}: {e}")
            continue
        df.insert(0, "source_file", str(path.relative_to(folder)))
        frames.append(df)
    if not frames:
        raise ValueError(f"none of the {len(files)} files in {folder} could be loaded")
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--recursive", action="store_true", help="include subfolders")
    parser.add_argument("--record-path", help="key holding the list of records in each file")
    parser.add_argument("--meta", nargs="+", help="top-level keys to copy onto each record")
    parser.add_argument("--out", type=Path, help="save to .csv, .pkl or .parquet")
    args = parser.parse_args()

    df = load_json_folder(args.folder, args.pattern, args.recursive, args.record_path, args.meta)
    print(f"{len(df)} rows x {df.shape[1]} columns from {df['source_file'].nunique()} files")
    print(df.head())
    if args.out:
        suffix = args.out.suffix.lower()
        if suffix == ".pkl":
            df.to_pickle(args.out)
        elif suffix == ".parquet":
            df.to_parquet(args.out, index=False)
        else:
            df.to_csv(args.out, index=False)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
