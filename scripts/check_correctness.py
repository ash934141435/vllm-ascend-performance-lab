#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import urllib.request
from pathlib import Path


def request_json(url, payload):
    request = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def capture(base_url, model, prompt_file, output, config, max_tokens=128):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("/model", trust_remote_code=True)
    rows = []
    for index, line in enumerate(Path(prompt_file).read_text().splitlines()):
        prompt = json.loads(line)["prompt"]
        request_id = f"{index:03d}-{hashlib.sha256(prompt.encode()).hexdigest()[:12]}"
        result = request_json(
            f"{base_url}/v1/completions",
            {"model": model, "prompt": prompt, "max_tokens": max_tokens,
             "temperature": 0, "seed": 0},
        )
        text = result["choices"][0]["text"]
        rows.append({
            "request_id": request_id,
            "config": config,
            "finish_reason": result["choices"][0].get("finish_reason", ""),
            "token_ids": tokenizer.encode(text, add_special_tokens=False),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        })
    Path(output).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    if any(row["finish_reason"] != "stop" for row in rows):
        raise SystemExit("correctness capture did not reach natural EOS")


def compare(reference_path, candidate_path, output_path):
    reference = {row["request_id"]: row for row in json.loads(Path(reference_path).read_text())}
    candidate = {row["request_id"]: row for row in json.loads(Path(candidate_path).read_text())}
    request_ids = sorted(set(reference) | set(candidate))
    rows = []
    for request_id in request_ids:
        left, right = reference.get(request_id), candidate.get(request_id)
        left_ids = left["token_ids"] if left else []
        right_ids = right["token_ids"] if right else []
        divergence = next(
            (index for index, pair in enumerate(zip(left_ids, right_ids)) if pair[0] != pair[1]),
            min(len(left_ids), len(right_ids)) if left_ids != right_ids else "",
        )
        rows.append({
            "request_id": request_id,
            "reference_config": left["config"] if left else "missing",
            "candidate_config": right["config"] if right else "missing",
            "exact_token_ids": left_ids == right_ids and left is not None and right is not None,
            "first_divergence": divergence,
            "reference_tokens": len(left_ids),
            "candidate_tokens": len(right_ids),
            "reference_finish_reason": left.get("finish_reason", "") if left else "missing",
            "candidate_finish_reason": right.get("finish_reason", "") if right else "missing",
        })
    with Path(output_path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--base-url", default="http://127.0.0.1:8010")
    cap.add_argument("--model", default="Qwen3.6-27B")
    cap.add_argument("--prompt-file", required=True)
    cap.add_argument("--output", required=True)
    cap.add_argument("--config", required=True)
    cap.add_argument("--max-tokens", type=int, default=128)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--reference", required=True)
    cmp.add_argument("--candidate", required=True)
    cmp.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.base_url, args.model, args.prompt_file, args.output, args.config, args.max_tokens)
    else:
        rows = compare(args.reference, args.candidate, args.output)
        if not all(row["exact_token_ids"] for row in rows):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
