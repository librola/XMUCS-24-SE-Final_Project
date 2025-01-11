from pathlib import Path
from aglibro.util.utils import load_jsonl
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rerank_dir", type=str, default="results/rerank", required=True)
parser.add_argument("--edits_file", type=str, default="success_patches.jsonl")
parser.add_argument("--output_pre", type=str, default="all_preds")
parser.add_argument("--top_n", type=int, default=1)
args = parser.parse_args()

rerank_dir = Path(args.rerank_dir)
edits_path = rerank_dir / args.edits_file
output_pre = args.output_pre
top_n = args.top_n
all_edits = load_jsonl(edits_path)

for i in range(top_n):
    if i or top_n > 1:
        file_name = rerank_dir / f"{output_pre}-{i}.jsonl"
    else:
        file_name = rerank_dir / f"{output_pre}.jsonl"
    with open(file_name, "w") as f:
        pass

    for edits in all_edits:
        with open(file_name, "a") as f:
            f.write(json.dumps({
                "model_name_or_path": "aglibro",
                "instance_id": edits['instance_id'],
                "model_patch": edits['edits'][i]['model_patch'] if edits['edits'] and i < len(edits['edits']) else "",
            }) + "\n")
