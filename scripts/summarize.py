import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import BENCHMARKS, parse_letter

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = BENCHMARKS[:4]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser(description="Accuracy per benchmark, 4-public and 6-benchmark unweighted averages, tokens and cost.")
    ap.add_argument("--run-id", default="main")
    ap.add_argument("--config", default=str(ROOT / "model_config.json"))
    args = ap.parse_args()
    config = json.load(open(args.config))
    nopt = {b: {json.loads(l)["id"]: len(json.loads(l)["options"]) for l in open(ROOT / "data" / f"{b}.jsonl")} for b in BENCHMARKS}
    cols = ["Model"] + BENCHMARKS + ["avg4_public", "avg6", "out_tokens", "cost_cny"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for mdir in sorted(p for p in (ROOT / "runs" / args.run_id).iterdir() if p.is_dir()):
        acc, out_tok, in_tok = {}, 0, 0
        for b in BENCHMARKS:
            path = mdir / f"{b}.jsonl"
            if not path.exists():
                continue
            rs = [json.loads(l) for l in open(path)]
            ok = 0
            for r in rs:
                if "error" in r:
                    continue
                in_tok += r["usage"].get("prompt_tokens", 0)
                out_tok += r["usage"].get("completion_tokens", 0)
                if r.get("finish_reason") != "length":
                    ok += parse_letter(r.get("content"), nopt[b][r["id"]]) == r["answer"]
            acc[b] = 100 * ok / len(rs)
        a4 = mean([acc[b] for b in PUBLIC if b in acc])
        a6 = mean(list(acc.values()))
        price = (config.get(mdir.name) or {}).get("price_cny_per_mtok") or {}
        cost = (in_tok * price["input"] + out_tok * price["output"]) / 1e6 if price else None
        cells = [mdir.name] + [f"{acc[b]:.1f}" if b in acc else "—" for b in BENCHMARKS]
        cells += [f"{a4:.1f}", f"{a6:.1f}", f"{out_tok:,}", f"{cost:.2f}" if cost is not None else "—"]
        lines.append("| " + " | ".join(cells) + " |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
