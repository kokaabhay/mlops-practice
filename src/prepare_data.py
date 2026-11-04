"""Downloads LEDGAR from LexGLUE and writes CSVs to data/raw."""
from datasets import load_dataset

from .config import ROOT, load_config


def main():
    cfg = load_config()
    out_dir = ROOT / cfg["data"]["raw_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    label_names = ds["train"].features["label"].names

    for split in ["train", "validation", "test"]:
        df = ds[split].to_pandas()
        df["label"] = df["label"].map(lambda i: label_names[i])
        df[["text", "label"]].to_csv(out_dir / f"ledgar_{split}.csv", index=False)
        print(f"{split}: {len(df)} rows")


if __name__ == "__main__":
    main()