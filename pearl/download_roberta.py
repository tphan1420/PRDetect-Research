"""Optional one-time download; never overwrites an existing local backbone."""
import argparse
from pathlib import Path


if __name__ == "__main__":
    from transformers import RobertaModel, RobertaTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/pearl/roberta-base")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Reuse the existing model directory: {path}")
    cache = path.parent / "hf_cache"
    model = RobertaModel.from_pretrained("roberta-base", revision=args.revision, cache_dir=str(cache))
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", revision=args.revision, cache_dir=str(cache), do_lower_case=False)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved frozen roberta-base to {path}")
