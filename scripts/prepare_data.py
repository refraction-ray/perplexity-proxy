"""Download the six benchmarks from Hugging Face and build data/<benchmark>.jsonl.

Public benchmarks are sampled with a fixed seed (<=300 items each).
BrowseComp and SimpleQA-verified are joined with our generated options in data/*_options.jsonl;
the correct letter is recovered from the official answers and never stored in this repo.
"""
import argparse
import base64
import hashlib
import json
import os
import random
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
SEED = 20261125
N = 300
LETTERS = "ABCDEFGHIJKLMNOP"
FILES = {
    "mmlu_pro.parquet": "datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet",
    "gpqa_diamond_mc.parquet": "datasets/hendrydong/gpqa_diamond_mc/resolve/main/data/test-00000-of-00001.parquet",
    "medx_text.jsonl": "datasets/TsinghuaC3I/MedXpertQA/resolve/main/Text/test.jsonl",
    "medx_mm.jsonl": "datasets/TsinghuaC3I/MedXpertQA/resolve/main/MM/test.jsonl",
    "medx_images.zip": "datasets/TsinghuaC3I/MedXpertQA/resolve/main/images.zip",
    "browse_comp.parquet": "datasets/smolagents/browse_comp/resolve/main/data/test-00000-of-00001.parquet",
    "simpleqa_verified.csv": "datasets/google/simpleqa-verified/resolve/main/simpleqa_verified.csv",
}
ANNOT_RE = re.compile(r"\s*\((?:acceptable range|aka\b|also know|or\b)[^()]*(?:\([^()]*\)[^()]*)*\)", re.I)


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, path in FILES.items():
        dst = RAW / name
        if not dst.exists():
            print("download", name)
            urllib.request.urlretrieve(f"{HF}/{path}", dst)
    if not (RAW / "images").exists():
        with zipfile.ZipFile(RAW / "medx_images.zip") as z:
            z.extractall(RAW)


def sample(rows, name):
    if len(rows) <= N:
        return rows
    rng = random.Random(f"{SEED}-{name}")
    return [rows[i] for i in sorted(rng.sample(range(len(rows)), N))]


def write(name, items):
    with open(DATA / f"{name}.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(name, len(items))


def mmlu_pro():
    rows = pd.read_parquet(RAW / "mmlu_pro.parquet").sort_values("question_id").to_dict("records")
    write("mmlu_pro", [{
        "id": f"mmlu_pro-{r['question_id']}", "benchmark": "mmlu_pro", "question": r["question"],
        "options": [str(o) for o in r["options"]], "answer": LETTERS[int(r["answer_index"])],
    } for r in sample(rows, "mmlu_pro")])


def gpqa_diamond():
    items = []
    for i, r in enumerate(pd.read_parquet(RAW / "gpqa_diamond_mc.parquet").to_dict("records")):
        body = r["problem"].partition("\n\nPlease write your final answer")[0]
        m = re.search(r"\n\(A\) (.*)\n\(B\) (.*)\n\(C\) (.*)\n\(D\) (.*)\s*$", body, re.S)
        items.append({
            "id": f"gpqa_diamond-{i}", "benchmark": "gpqa_diamond", "question": body[: m.start()].strip(),
            "options": [g.strip() for g in m.groups()], "answer": r["solution"].replace("\\boxed{", "").rstrip("}"),
        })
    write("gpqa_diamond", items)


def medxpertqa(split):
    name = f"medxpertqa_{split}"
    rows = [json.loads(l) for l in open(RAW / f"medx_{split}.jsonl")]
    items = []
    for r in sample(rows, name):
        keys = sorted(r["options"])
        it = {"id": r["id"], "benchmark": name, "question": r["question"].split("\nAnswer Choices:")[0].strip(),
              "options": [r["options"][k] for k in keys], "answer": r["label"]}
        if split == "mm":
            it["images"] = [f"raw/images/{p}" for p in r["images"]]
        items.append(it)
    write(name, items)


def decrypt(ciphertext_b64, password):
    enc = base64.b64decode(ciphertext_b64)
    key = hashlib.sha256(password.encode()).digest()
    key = (key * (len(enc) // len(key) + 1))[: len(enc)]
    return bytes(a ^ b for a, b in zip(enc, key)).decode()


def clean(s):
    s = re.sub(r"\s+", " ", ANNOT_RE.sub("", str(s))).strip()
    if len(s) > 2 and s[0] in "\"“'" and s[-1] in "\"”'":
        s = s[1:-1].strip()
    if s.endswith(".") and not re.search(r"\b[A-Z]\.$|\b(?:Jr|Sr|St|Co|Inc|Ltd)\.$", s):
        s = s[:-1].rstrip()
    return s.casefold()


def selfbuilt(name, source):
    items = []
    for l in open(DATA / f"{name}_options.jsonl"):
        r = json.loads(l)
        question, gold = source[r["source_id"]]
        hits = [i for i, o in enumerate(r["options"]) if clean(o) == clean(gold)]
        assert len(hits) == 1, (name, r["source_id"])
        items.append({"id": f"{name}-{r['source_id']}", "benchmark": name, "question": question,
                      "options": r["options"], "answer": LETTERS[hits[0]]})
    write(name, items)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    download()
    mmlu_pro()
    gpqa_diamond()
    medxpertqa("text")
    medxpertqa("mm")
    bc = pd.read_parquet(RAW / "browse_comp.parquet").reset_index(drop=True)
    selfbuilt("browsecomp_mcq", {i: (decrypt(r["problem"], r["canary"]), decrypt(r["answer"], r["canary"])) for i, r in bc.iterrows()})
    sq = pd.read_csv(RAW / "simpleqa_verified.csv")
    selfbuilt("simpleqa_verified_mcq", {int(r["original_index"]): (r["problem"], str(r["answer"])) for _, r in sq.iterrows()})


if __name__ == "__main__":
    main()
