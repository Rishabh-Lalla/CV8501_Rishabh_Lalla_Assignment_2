import argparse, os, json
import pandas as pd
from ..constants import LABELS

QUESTION = "What is the most likely diagnosis? Answer with one token from: [akiec, bcc, bkl, df, mel, nv, vasc]."

def convert_split(root: str, split_csv: str, out_jsonl: str):
    df = pd.read_csv(split_csv)
    images_dir = os.path.join(root, "images")
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for _, row in df.iterrows():
            img_path = os.path.join(images_dir, f"{row['image_id']}.jpg")
            ex = {
                "image_path": img_path,
                "question": QUESTION,
                "answer": str(row["dx"]).strip(),
            }
            out.write(json.dumps(ex) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="HAM10000 root with images/ and splits/*.csv")
    ap.add_argument("--out", type=str, required=True, help="Output dir for jsonl files")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for split in ["train", "val", "test"]:
        convert_split(args.root, os.path.join(args.root, "splits", f"{split}.csv"),
                      os.path.join(args.out, f"{split}.jsonl"))
    print(f"Wrote VQA jsonl files to {args.out}")

if __name__ == "__main__":
    main()
