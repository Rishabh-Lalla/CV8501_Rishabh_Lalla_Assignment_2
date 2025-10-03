import argparse, os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from typing import Tuple
import numpy as np

def make_splits(meta_csv: str, seed: int, out_dir: str, train_frac=0.7, val_frac=0.1, test_frac=0.2):
    df = pd.read_csv(meta_csv)
    needed = {"image_id", "dx", "lesion_id"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(f"Metadata missing columns: need {needed}")
    labels = df["dx"].values
    groups = df["lesion_id"].values

    # First get test split (20%)
    sgkf = StratifiedGroupKFold(n_splits=int(1/test_frac), shuffle=True, random_state=seed)
    test_idx = next(sgkf.split(df, labels, groups))[1]
    df_test = df.iloc[test_idx].copy()
    df_rem = df.drop(df.index[test_idx]).copy()

    # Now split remaining into train/val (approx 70/10)
    rem_labels = df_rem["dx"].values
    rem_groups = df_rem["lesion_id"].values
    val_ratio = val_frac / (train_frac + val_frac)
    n_splits = int(1/val_ratio)
    sgkf2 = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf2.split(df_rem, rem_labels, rem_groups))
    df_val = df_rem.iloc[val_idx].copy()
    df_train = df_rem.drop(df_rem.index[val_idx]).copy()

    return df_train, df_val, df_test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Path containing images/ and HAM10000_metadata.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    meta_csv = os.path.join(args.root, "HAM10000_metadata.csv")
    out_dir = os.path.join(args.root, "splits")
    os.makedirs(out_dir, exist_ok=True)
    train, val, test = make_splits(meta_csv, args.seed, out_dir)

    for name, df in [("train", train), ("val", val), ("test", test)]:
        df[["image_id", "dx", "lesion_id"]].to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    print(f"Wrote splits to {out_dir}")


if __name__ == "__main__":
    main()
