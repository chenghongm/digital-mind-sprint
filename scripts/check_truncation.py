"""
统计 runs/v2/meta/*.json 中 model_text 的 token 数，
计算 token 数 > MAX_NEW_TOKENS(60) 的比例。

生成时 MAX_NEW_TOKENS=60 是硬截断上限，所以正常情况下
用同一个 tokenizer 数出来的 token 数不会超过 60。
这里统计的是 token 数 >= 60（即打满截断上限、没有自然结束）的占比，
用来估计有多少输出是被截断的。
"""

import argparse
import glob
import json
import os

from transformers import AutoTokenizer

MAX_NEW_TOKENS = 60
DEFAULT_TOKENIZER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Llama-3.1-8B-Instruct"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "runs", "v2", "meta"),
        help="包含 *.json 文件的目录",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER_PATH,
        help="tokenizer 路径或 HF 模型名",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=MAX_NEW_TOKENS,
        help="判定为“打满截断上限”的 token 数阈值",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    files = sorted(glob.glob(os.path.join(args.meta_dir, "*.json")))
    if not files:
        raise SystemExit(f"未找到 json 文件: {args.meta_dir}")

    total = 0
    truncated = 0
    per_file_rows = []

    for path in files:
        with open(path) as f:
            data = json.load(f)

        file_total = 0
        file_truncated = 0

        for turn in data.get("turns", []):
            text = turn.get("model_text", "")
            n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            file_total += 1
            if n_tokens >= args.threshold:
                file_truncated += 1

        total += file_total
        truncated += file_truncated
        per_file_rows.append(
            (os.path.basename(path), file_truncated, file_total)
        )

    print(f"{'file':<35} {'truncated':>10} {'total':>8} {'ratio':>8}")
    for name, t, n in per_file_rows:
        ratio = t / n if n else 0.0
        print(f"{name:<35} {t:>10} {n:>8} {ratio:>8.2%}")

    overall_ratio = truncated / total if total else 0.0
    print("-" * 65)
    print(f"{'TOTAL':<35} {truncated:>10} {total:>8} {overall_ratio:>8.2%}")


if __name__ == "__main__":
    main()
