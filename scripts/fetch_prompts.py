#!/usr/bin/env python3
"""Freeze a fixed, public, real-text prompt dataset for stage-2 ablation.

Pulls real text from a pinned public corpus, tokenizes with the EXACT served
model tokenizer, bins prompts into the prefill-heavy (~4096 tok) and
decode-heavy (~1024 tok) workloads, and writes:

  <out-dir>/prefill_heavy.jsonl   # [{"prompt": "..."}, ...]
  <out-dir>/decode_heavy.jsonl    # [{"prompt": "..."}, ...]
  <out-dir>/manifest.json         # dataset/revision/rules + per-file SHA256 + token counts

No synthetic/random tokens. Every prompt is real public text; its true token
count is recorded so MTP acceptance is never explained by random tokens.

The frozen files are the deliverable; this script need only re-run if the
corpus, tokenizer, or selection rules change.
"""
import argparse
import hashlib
import json
import os
import random

try:
    from tokenizers import Tokenizer
except ImportError:  # server container may only expose it via transformers
    from transformers import AutoTokenizer

    def _load(path):
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    Tokenizer = None
else:
    def _load(path):
        return Tokenizer.from_file(path)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def token_count(tok, text: str) -> int:
    if Tokenizer is not None:
        return len(tok.encode(text).ids)
    return len(tok.encode(text))


def select_nearest(items, target, window, num, seed):
    """items: list of (text, ntok). Return up to num nearest to target."""
    items = [it for it in items if abs(it[1] - target) <= window]
    items.sort(key=lambda it: abs(it[1] - target))
    return items[:num]


def select_stitched(items, target, lo, hi, num, seed):
    """Build `num` prompts near `target` by concatenating real texts.

    Each prompt = one base text (>=1800 tok) + shortest available fillers
    ([200,1600] tok) until landing in [lo, hi]. All pieces are real public
    text; the combined token count is exact per the served tokenizer. This
    is needed because the pinned corpus (ultrachat_200k) has no single texts
    near 4096 tokens, so we stitch adjacent real texts to hit the prefill
    target. Returns list of (text, ntok).
    """
    random.seed(seed)
    base = sorted([s for s in items if s[1] >= 1800], key=lambda x: -x[1])
    fillers = sorted([s for s in items if 200 <= s[1] <= 1600], key=lambda x: x[1])
    random.shuffle(base)
    out = []
    used_f = set()
    for b in base:
        if len(out) >= num:
            break
        pieces = [b[0]]
        cur = b[1]
        for fi, f in enumerate(fillers):
            if fi in used_f:
                continue
            if cur + f[1] <= hi:
                cur += f[1]
                pieces.append(f[0])
                used_f.add(fi)
                if cur >= lo:
                    break
        if lo <= cur <= hi + 200:
            out.append(("\n\n".join(pieces), cur))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="JSON file: list of prompt strings")
    ap.add_argument("--tokenizer", required=True, help="path to model tokenizer.json")
    ap.add_argument("--out-dir", default="results/execution_ablation/prompts")
    ap.add_argument("--dataset-name", default="HuggingFaceH4/ultrachat_200k")
    ap.add_argument("--dataset-revision", default="train_sft shard 0")
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--prefill-target", type=int, default=4096)
    ap.add_argument("--decode-target", type=int, default=1024)
    ap.add_argument("--prefill-window", type=int, default=700)
    ap.add_argument("--decode-window", type=int, default=350)
    ap.add_argument("--prefill-stitch", action="store_true",
                    help="stitch real texts to reach ~4096 (corpus has no single 4096-tok text)")
    ap.add_argument("--prefill-lo", type=int, default=3700)
    ap.add_argument("--prefill-hi", type=int, default=4500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    tok = _load(args.tokenizer)

    texts = json.load(open(args.corpus, encoding="utf-8"))
    items = []
    for t in texts:
        if not isinstance(t, str) or len(t) < 50:
            continue
        items.append((t, token_count(tok, t)))
    items.sort(key=lambda it: random.random())  # stable shuffle before selection

    if args.prefill_stitch:
        prefill = select_stitched(items, args.prefill_target, args.prefill_lo, args.prefill_hi, args.num_prompts, args.seed)
    else:
        prefill = select_nearest(items, args.prefill_target, args.prefill_window, args.num_prompts, args.seed)
    decode = select_nearest(items, args.decode_target, args.decode_window, args.num_prompts, args.seed)

    # Relax windows if a bucket is short (keep real text, widen acceptance).
    for _ in range(6):
        if len(prefill) < args.num_prompts:
            args.prefill_window += 300
            prefill = select_nearest(items, args.prefill_target, args.prefill_window, args.num_prompts, args.seed)
        if len(decode) < args.num_prompts:
            args.decode_window += 200
            decode = select_nearest(items, args.decode_target, args.decode_window, args.num_prompts, args.seed)
        if len(prefill) >= args.num_prompts and len(decode) >= args.num_prompts:
            break

    def write_workload(name, sel):
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for text, _ in sel:
                f.write(json.dumps({"prompt": text}, ensure_ascii=False) + "\n")
        return path, sha256_file(path), [n for _, n in sel]

    pf_path, pf_sha, pf_tok = write_workload("prefill_heavy", prefill)
    dc_path, dc_sha, dc_tok = write_workload("decode_heavy", decode)

    tok_sha = sha256_file(args.tokenizer)
    manifest = {
        "dataset_name": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "tokenizer_sha256": tok_sha,
        "selection_rules": {
            "num_prompts_per_workload": args.num_prompts,
            "prefill_target_tokens": args.prefill_target,
            "decode_target_tokens": args.decode_target,
            "prefill_stitch": args.prefill_stitch,
            "prefill_window": args.prefill_window,
            "prefill_lo": args.prefill_lo,
            "prefill_hi": args.prefill_hi,
            "decode_window": args.decode_window,
            "seed": args.seed,
            "note": "real public text (ultrachat_200k); stitched prefill prompts are concatenations of real public texts; token counts are exact per served tokenizer",
        },
        "workloads": {
            "prefill_heavy": {
                "file": os.path.basename(pf_path),
                "sha256": pf_sha,
                "count": len(pf_tok),
                "token_counts": pf_tok,
            },
            "decode_heavy": {
                "file": os.path.basename(dc_path),
                "sha256": dc_sha,
                "count": len(dc_tok),
                "token_counts": dc_tok,
            },
        },
    }
    mpath = os.path.join(args.out_dir, "manifest.json")
    json.dump(manifest, open(mpath, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"prefill_heavy: {len(pf_tok)} prompts, tokens min/med/max="
          f"{min(pf_tok)}/{sorted(pf_tok)[len(pf_tok)//2]}/{max(pf_tok)} sha={pf_sha[:12]}")
    print(f"decode_heavy:  {len(dc_tok)} prompts, tokens min/med/max="
          f"{min(dc_tok)}/{sorted(dc_tok)[len(dc_tok)//2]}/{max(dc_tok)} sha={dc_sha[:12]}")
    print(f"manifest -> {mpath}")


if __name__ == "__main__":
    main()
