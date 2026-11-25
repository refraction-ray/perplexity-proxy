import argparse
import base64
import json
import mimetypes
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BENCHMARKS = [
    "mmlu_pro",
    "gpqa_diamond",
    "medxpertqa_text",
    "medxpertqa_mm",
    "browsecomp_mcq",
    "simpleqa_verified_mcq",
]
LETTERS = "ABCDEFGHIJKLMNOP"
SYSTEM = (
    "The following is a multiple choice question. "
    "Respond with only the letter of the correct answer, and nothing else."
)


def load_items(bench, limit):
    items = [json.loads(l) for l in open(DATA / f"{bench}.jsonl")]
    return items[:limit] if limit else items


def sniff_mime(data, name):
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return mimetypes.guess_type(name)[0] or "image/jpeg"


def image_part(rel):
    path = DATA / rel
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{sniff_mime(data, path.name)};base64,{b64}"}}


def build_messages(item):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(item["options"]))
    text = f"{item['question']}\n\n{opts}\n\nAnswer:"
    if item.get("images"):
        content = [image_part(p) for p in item["images"]] + [{"type": "text", "text": text}]
    else:
        content = text
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def parse_letter(text, n):
    valid = LETTERS[:n]
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    m = re.fullmatch(r"[\s\*\(\[]*([A-P])[\s\*\)\]\.:]*", text)
    if m and m.group(1) in valid:
        return m.group(1)
    lead = re.match(r"[\s\*#>\(\[]*(?:(?:final\s+)?answer\s*(?:is)?\s*[:：]?\s*)?[\s\*\(\[]*([A-P])(?=\s*$|[\.\):：\*\],\n]|\s+[—–-]\s)", text, flags=re.I)
    if lead and lead.group(1) in valid:
        return lead.group(1)
    patterns = [
        r"\\boxed\{\s*([A-P])\s*\}",
        r"(?:[Aa]nswer|ANSWER)\s*(?:is)?\s*[:：]?\s*\**\(?([A-P])\b",
        r"\*\*\(?([A-P])\)?\*\*",
        r"\*\*\(?([A-P])[\.\)]\s",
        r"^\(?([A-P])[\.\):]",
        r"(?:^|[\.\s\*])([A-P])[\.\*\s]*\Z",
    ]
    for pat in patterns:
        hits = [h for h in re.findall(pat, text, flags=re.M) if h in valid]
        if hits:
            return hits[-1]
    return None


def usage_dict(u):
    if u is None:
        return {}
    d = u.model_dump() if hasattr(u, "model_dump") else dict(u)
    details = d.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": d.get("prompt_tokens") or 0,
        "completion_tokens": d.get("completion_tokens") or 0,
        "reasoning_tokens": details.get("reasoning_tokens") or 0,
        "cached_tokens": (d.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
    }


def call(client, model, cfg, item, max_tokens, retries=6):
    kwargs = dict(model=model, messages=build_messages(item), max_tokens=max_tokens)
    if cfg.get("temperature", 0) is not None:
        kwargs["temperature"] = cfg.get("temperature", 0)
    kwargs.update(cfg.get("top_level") or {})
    if cfg.get("extra_body"):
        kwargs["extra_body"] = cfg["extra_body"]
    last = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = client.chat.completions.create(**kwargs)
            msg = r.choices[0].message
            extra = msg.model_extra or {}
            reasoning = extra.get("reasoning_content") or extra.get("reasoning") or ""
            content = msg.content or ""
            pred = parse_letter(content, len(item["options"]))
            return {
                "id": item["id"],
                "benchmark": item["benchmark"],
                "answer": item["answer"],
                "pred": pred,
                "correct": pred == item["answer"],
                "content": content,
                "reasoning_chars": len(reasoning),
                "finish_reason": r.choices[0].finish_reason,
                "usage": usage_dict(r.usage),
                "latency_s": round(time.time() - t0, 2),
                "attempts": attempt + 1,
            }
        except Exception as e:
            last = e
            status = getattr(e, "status_code", None)
            if status in (400, 404) and attempt >= 1:
                break
            time.sleep(min(60, 2 ** attempt * 2))
    return {
        "id": item["id"],
        "benchmark": item["benchmark"],
        "answer": item["answer"],
        "pred": None,
        "correct": False,
        "error": f"{type(last).__name__}: {str(last)[:500]}",
        "usage": {},
    }


def run_model(client, model, cfg, benches, args):
    out_dir = ROOT / "runs" / args.run_id / model
    out_dir.mkdir(parents=True, exist_ok=True)
    for bench in benches:
        if bench == "medxpertqa_mm" and not cfg.get("vision", False):
            print(f"[{model}] skip {bench}: no vision")
            continue
        bcfg = {**cfg, **(cfg.get("overrides", {}).get(bench) or {})}
        path = out_dir / f"{bench}.jsonl"
        done = {}
        if path.exists():
            for l in open(path):
                r = json.loads(l)
                if "error" not in r and not (args.retry_truncated and r.get("finish_reason") == "length"):
                    done[r["id"]] = r
        items = [it for it in load_items(bench, args.limit) if it["id"] not in done]
        if path.exists():
            with open(path, "w") as f:
                for r in done.values():
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{model}] {bench}: {len(done)} cached, {len(items)} to run", flush=True)
        if not items:
            continue
        lock = threading.Lock()
        n_ok = n_err = 0
        with open(path, "a") as f, ThreadPoolExecutor(args.workers) as ex:
            futs = [ex.submit(call, client, model, bcfg, it, bcfg.get("max_tokens", args.max_tokens)) for it in items]
            for fut in as_completed(futs):
                r = fut.result()
                with lock:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    if "error" in r:
                        n_err += 1
                    else:
                        n_ok += 1
        print(f"[{model}] {bench}: done ok={n_ok} err={n_err}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=BENCHMARKS)
    ap.add_argument("--run-id", default="main")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--retry-truncated", action="store_true")
    ap.add_argument("--config", default=str(ROOT / "model_config.json"))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"), help="any OpenAI-compatible endpoint, e.g. https://api.example.com/v1")
    args = ap.parse_args()

    configs = json.load(open(args.config))
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=args.base_url, timeout=600, max_retries=0)
    with ThreadPoolExecutor(len(args.models)) as ex:
        futs = [ex.submit(run_model, client, m, configs.get(m, {}), args.benchmarks, args) for m in args.models]
        for fut in futs:
            fut.result()


if __name__ == "__main__":
    main()
