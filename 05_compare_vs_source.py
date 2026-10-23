#!/usr/bin/env python3
"""
05_compare_vs_source.py

Compare Gemma and Qwen summaries against the SOURCE dialogue (no ground truth).

Input : two folders of <movie>_output.json (interval / preprocessed_input / summary)
Output: <out_dir>/compare_vs_source.csv  - per-chunk scores + per-chunk winner
        <out_dir>/compare_vs_source.md   - averages + overall verdict

Metrics (summary vs source)
---------------------------
faithfulness  NLI: mean entailment prob of each summary sentence given the
              dialogue (sliding windows). Higher = fewer hallucinations.   [main]
semantic_sim  cosine similarity of sentence embeddings (summary vs dialogue).
              Higher = summary is about what the dialogue is about.
rouge1_p / rouge2_p / rougeL_p
              ROUGE PRECISION vs source = share of summary words / bigrams /
              sequence found in the dialogue. Higher = more grounded wording.
              (Recall vs a long source is meaningless for a 2-3 sentence summary.)
novel_names   capitalised names in the summary not present in the dialogue.
words, compression (summary words / source words)

Per-chunk winner: higher faithfulness; if within 0.05, higher semantic_sim.

Example
-------
python 05_compare_vs_source.py --gemma ./summaries_gemma --qwen ./summaries_qwen -o ./compare
"""

import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path

SENT_RE = re.compile(r"(?<=[.!?])\s+")
NOT_NAMES = set("""
I A An The He She They It We You His Her Their Its Our Your This That These Those There Here Then
When While After Before As At In On Of For With Without From To Into Meanwhile However Later Finally
Mr Mrs Ms Dr Sir Madam Mom Mum Dad Mother Father God Lord Okay OK Yes No
""".split())


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load(folder):
    data = {}
    for fp in sorted(Path(folder).glob("*_output.json")):
        movie = re.sub(r"_output$", "", fp.stem)
        for r in json.load(open(fp, encoding="utf-8")):
            s = (r.get("summary") or "").strip()
            if s:
                data[(movie, r["interval"])] = (r.get("preprocessed_input", ""), s)
    return data


# --------------------------------------------------------------------------
# Lexical metrics
# --------------------------------------------------------------------------
def toks(t):
    return re.findall(r"[a-z0-9']+", t.lower())


def sentences(t):
    return [s for s in SENT_RE.split(t.strip()) if s.strip()]


def ngram_precision(summary, source, n):
    s, src = toks(summary), toks(source)
    sg = [tuple(s[i:i + n]) for i in range(len(s) - n + 1)]
    if not sg:
        return 0.0
    pool = {}
    for g in (tuple(src[i:i + n]) for i in range(len(src) - n + 1)):
        pool[g] = pool.get(g, 0) + 1
    hit = 0
    for g in sg:
        if pool.get(g, 0) > 0:
            pool[g] -= 1
            hit += 1
    return hit / len(sg)


def lcs_precision(summary, source):
    a, b = toks(summary), toks(source)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1] / len(a)


def novel_names(summary, source):
    src, out = source.lower(), set()
    for sent in sentences(summary):
        words = re.findall(r"[A-Za-z][A-Za-z'\-]+", sent)
        for i, w in enumerate(words):
            base = re.sub(r"'s?$", "", w)
            if i and base[:1].isupper() and base not in NOT_NAMES and base.lower() not in src:
                out.add(base)
    return sorted(out)


# --------------------------------------------------------------------------
# Model-based metrics (optional)
# --------------------------------------------------------------------------
class NLI:
    def __init__(self, name):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForSequenceClassification.from_pretrained(name).eval()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m.to(self.dev)
        lab = {v.lower(): k for k, v in self.m.config.id2label.items()}
        self.ent = lab.get("entailment", 1)

    def score(self, source, summary, size=300, stride=200):
        w = source.split()
        wins = [source] if len(w) <= size else \
            [" ".join(w[i:i + size]) for i in range(0, len(w) - size + stride, stride)]
        sents = sentences(summary) or [summary]
        pairs = [(win, s) for s in sents for win in wins]
        probs = []
        for i in range(0, len(pairs), 16):
            b = pairs[i:i + 16]
            enc = self.tok([p for p, _ in b], [h for _, h in b], truncation="only_first",
                           max_length=512, padding=True, return_tensors="pt").to(self.dev)
            with self.torch.no_grad():
                probs += self.m(**enc).logits.softmax(-1)[:, self.ent].tolist()
        per = [max(probs[k * len(wins):(k + 1) * len(wins)]) for k in range(len(sents))]
        return sum(per) / len(per)


class Embed:
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)

    def sim(self, a, b):
        from sentence_transformers import util
        e = self.m.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
        return float(util.cos_sim(e[0], e[1]))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
METRICS = [  # key, label, higher_is_better
    ("faithfulness", "NLI faithfulness (main)", True),
    ("semantic_sim", "Semantic similarity to source", True),
    ("rouge1_p", "ROUGE-1 precision vs source", True),
    ("rouge2_p", "ROUGE-2 precision vs source", True),
    ("rougeL_p", "ROUGE-L precision vs source", True),
    ("novel_name_rate", "Chunks with names not in source", False),
    ("words", "Avg words", None),
    ("compression", "Compression (summary/source words)", None),
]


def main():
    ap = argparse.ArgumentParser(description="Compare Gemma vs Qwen summaries against the source dialogue.")
    ap.add_argument("--gemma", required=True, help="Folder with Gemma *_output.json")
    ap.add_argument("--qwen", required=True, help="Folder with Qwen *_output.json")
    ap.add_argument("-o", "--out_dir", default="./compare")
    ap.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-base", help="'' to skip")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2", help="'' to skip")
    args = ap.parse_args()

    runs = {"gemma": load(args.gemma), "qwen": load(args.qwen)}
    common = sorted(k for k in set(runs["gemma"]) & set(runs["qwen"])
                    if runs["gemma"][k][0] == runs["qwen"][k][0])
    print(f"gemma: {len(runs['gemma'])}  qwen: {len(runs['qwen'])}  common chunks (same input): {len(common)}")
    if not common:
        print("Nothing to compare. Run both models on the same *_chunk.json files.")
        return

    nli = emb = None
    if args.nli_model:
        try:
            nli = NLI(args.nli_model)
        except Exception as e:
            print(f"[WARN] NLI skipped: {e}")
    if args.embed_model:
        try:
            emb = Embed(args.embed_model)
        except Exception as e:
            print(f"[WARN] semantic similarity skipped: {e}")

    rows = []
    for i, key in enumerate(common):
        src = runs["gemma"][key][0]
        row = {"movie": key[0], "interval": key[1], "source_words": len(src.split()), "source": src}
        for m in ("gemma", "qwen"):
            s = runs[m][key][1]
            names = novel_names(s, src)
            row.update({
                f"{m}_summary": s,
                f"{m}_faithfulness": round(nli.score(src, s), 4) if nli else None,
                f"{m}_semantic_sim": round(emb.sim(s, src), 4) if emb else None,
                f"{m}_rouge1_p": round(ngram_precision(s, src, 1), 4),
                f"{m}_rouge2_p": round(ngram_precision(s, src, 2), 4),
                f"{m}_rougeL_p": round(lcs_precision(s, src), 4),
                f"{m}_novel_names": ", ".join(names),
                f"{m}_novel_name_rate": int(bool(names)),
                f"{m}_words": len(s.split()),
                f"{m}_compression": round(len(s.split()) / max(1, len(src.split())), 4),
            })
        # per-chunk winner
        fg, fq = row["gemma_faithfulness"], row["qwen_faithfulness"]
        sg, sq = row["gemma_semantic_sim"], row["qwen_semantic_sim"]
        if fg is not None and abs(fg - fq) > 0.05:
            row["winner"] = "gemma" if fg > fq else "qwen"
        elif sg is not None and abs(sg - sq) > 0.02:
            row["winner"] = "gemma" if sg > sq else "qwen"
        elif fg is None and sg is None:
            rg, rq = row["gemma_rougeL_p"], row["qwen_rougeL_p"]
            row["winner"] = "tie" if abs(rg - rq) <= 0.05 else ("gemma" if rg > rq else "qwen")
        else:
            row["winner"] = "tie"
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  scored {i + 1}/{len(common)}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "compare_vs_source.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def avg(m, k):
        v = [r[f"{m}_{k}"] for r in rows if r[f"{m}_{k}"] is not None]
        return st.mean(v) if v else None

    pts = {"gemma": 0, "qwen": 0}
    L = ["# Gemma vs Qwen — summaries compared to source dialogue", "",
         f"Common chunks: {len(rows)}", "",
         "| Metric | Gemma | Qwen | Better |", "|---|---|---|---|"]
    for k, label, hib in METRICS:
        g, q = avg("gemma", k), avg("qwen", k)
        if g is None:
            continue
        better = ""
        if hib is not None and abs(g - q) > 1e-4:
            better = "gemma" if (g > q) == hib else "qwen"
            pts[better] += 2 if k == "faithfulness" else 1
        pct = k in ("novel_name_rate", "compression")
        f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.3f}" if k != "words" else f"{x:.1f}")
        L.append(f"| {label} | {f(g)} | {f(q)} | {better} |")

    wins = {w: sum(r["winner"] == w for r in rows) for w in ("gemma", "qwen", "tie")}
    n = len(rows)
    L += ["", f"**Per-chunk winner:** Gemma {wins['gemma']} ({100 * wins['gemma'] / n:.1f}%), "
              f"Qwen {wins['qwen']} ({100 * wins['qwen'] / n:.1f}%), tie {wins['tie']} ({100 * wins['tie'] / n:.1f}%)"]
    if wins["gemma"] != wins["qwen"]:
        pts["gemma" if wins["gemma"] > wins["qwen"] else "qwen"] += 2
    best = max(pts, key=pts.get)
    L += ["", f"Points (faithfulness and per-chunk wins count 2): gemma={pts['gemma']}, qwen={pts['qwen']}", "",
          f"**Better vs source: {best}**" if pts["gemma"] != pts["qwen"] else "**No clear winner.**",
          "", "Note: these metrics measure grounding in the source, not how well the key events "
              "were chosen. Read ~20 rows of the CSV (sort by the faithfulness gap) to confirm."]
    report = "\n".join(L)
    (out / "compare_vs_source.md").write_text(report, encoding="utf-8")
    print("\n" + report + f"\n\nFiles: {out.resolve()}")


if __name__ == "__main__":
    main()
