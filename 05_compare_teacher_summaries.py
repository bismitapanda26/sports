#!/usr/bin/env python3
"""
05_compare_teacher_summaries.py

Compare teacher summaries from different models (e.g. Gemma vs Qwen) WITHOUT
ground truth, to choose which teacher to distil from.

Input : one output folder per model, as written by 03_gemma_summary_generator.py
        (<movie>_output.json with interval / preprocessed_input / summary).
Output: <out_dir>/compare_report.md     - overall verdict + metric table
        <out_dir>/compare_details.csv   - per-chunk scores for every model
        <out_dir>/human_review.csv      - blind A/B sheet for manual spot-check
        <out_dir>/human_review_key.csv  - which model is A / B (open after rating)

Signals (all reference-free)
----------------------------
1. Format     : words, sentences, compression, % within the 2-3 sentence / 50-word spec.
2. Grounding  : NLI faithfulness - each summary sentence must be entailed by the
                dialogue (DeBERTa NLI cross-encoder). Higher = fewer hallucinations.
3. Names      : capitalised names in the summary that never appear in the dialogue
                (cheap hallucination check, no model needed).
4. Judge      : optional pairwise LLM-as-judge on an OpenAI-compatible API, run in
                BOTH orders to cancel position bias. Use a judge that is NOT one of
                the compared models if possible.
5. Agreement  : ROUGE-1/ROUGE-L between models (context only, not a quality score).

Example
-------
python 05_compare_teacher_summaries.py \
    --runs gemma=./summaries_gemma qwen=./summaries_qwen -o ./compare \
    --judge_model gemini-2.5-pro --judge_url http://<server>:4000/v1/chat/completions \
    --judge_sample 200
"""

import argparse
import csv
import itertools
import json
import os
import random
import re
import statistics as st
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SPEC_MIN_SENT, SPEC_MAX_SENT, SPEC_MAX_WORDS = 2, 3, 50
SENT_RE = re.compile(r"(?<=[.!?])\s+")

# Capitalised words that are not names
NOT_NAMES = set("""
I A An The He She They It We You His Her Their Its Our Your This That These Those There Here Then
When While After Before As At In On Of For With Without From To Into Meanwhile However Later Finally
Mr Mrs Ms Dr Sir Madam Mom Mum Dad Mother Father God Lord Okay OK Yes No
Monday Tuesday Wednesday Thursday Friday Saturday Sunday Christmas
""".split())


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_run(folder):
    """{(movie, interval): {"text":..., "summary":...}}"""
    data = {}
    for fp in sorted(Path(folder).glob("*_output.json")):
        movie = re.sub(r"_output$", "", fp.stem)
        for r in json.load(open(fp, encoding="utf-8")):
            s = (r.get("summary") or r.get("reference_summary") or "").strip()
            if not s or r.get("status", "ok") != "ok":
                continue
            data[(movie, r["interval"])] = {
                "text": r.get("preprocessed_input") or r.get("source_text", ""),
                "summary": s,
            }
    return data


# --------------------------------------------------------------------------
# Cheap metrics
# --------------------------------------------------------------------------
def sentences(t):
    return [s for s in SENT_RE.split(t.strip()) if s.strip()]


def tokens(t):
    return re.findall(r"[a-z0-9']+", t.lower())


def unknown_names(summary, source):
    src = source.lower()
    names = set()
    for sent in sentences(summary):
        words = re.findall(r"[A-Za-z][A-Za-z'\-]+", sent)
        for i, w in enumerate(words):
            base = re.sub(r"'s?$", "", w)
            if i == 0 or not base[:1].isupper() or base in NOT_NAMES:
                continue
            if base.lower() not in src:
                names.add(base)
    return sorted(names)


def rouge1_f(a, b):
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    ca, cb = {}, {}
    for t in ta:
        ca[t] = ca.get(t, 0) + 1
    for t in tb:
        cb[t] = cb.get(t, 0) + 1
    ov = sum(min(v, cb.get(k, 0)) for k, v in ca.items())
    if not ov:
        return 0.0
    p, r = ov / len(ta), ov / len(tb)
    return 2 * p * r / (p + r)


def rougeL_f(a, b):
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    prev = [0] * (len(tb) + 1)
    for x in ta:
        cur = [0]
        for j, y in enumerate(tb):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if not lcs:
        return 0.0
    p, r = lcs / len(ta), lcs / len(tb)
    return 2 * p * r / (p + r)


# --------------------------------------------------------------------------
# NLI faithfulness
# --------------------------------------------------------------------------
class NLIScorer:
    def __init__(self, model_name, device=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.ent = labels.get("entailment", labels.get("entail", 1))

    def _windows(self, source, size=300, stride=200):
        w = source.split()
        if len(w) <= size:
            return [source]
        return [" ".join(w[i:i + size]) for i in range(0, max(1, len(w) - size + stride), stride)]

    def score(self, source, summary):
        """Mean over summary sentences of max entailment prob across source windows."""
        sents = sentences(summary) or [summary]
        wins = self._windows(source)
        pairs = [(w, s) for s in sents for w in wins]
        probs = []
        for i in range(0, len(pairs), 16):
            batch = pairs[i:i + 16]
            enc = self.tok([p for p, _ in batch], [h for _, h in batch], truncation="only_first",
                           max_length=512, padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                pr = self.model(**enc).logits.softmax(-1)[:, self.ent]
            probs += pr.tolist()
        per_sent = [max(probs[k * len(wins):(k + 1) * len(wins)]) for k in range(len(sents))]
        return sum(per_sent) / len(per_sent), sum(p >= 0.5 for p in per_sent) / len(per_sent)


# --------------------------------------------------------------------------
# LLM judge
# --------------------------------------------------------------------------
JUDGE_SYSTEM = """You are a strict evaluator of movie-dialogue summaries.
You get subtitle dialogue from one movie segment and two candidate summaries (A and B).
"Name:" at the start of a sentence is a speaker label; [music], [crying] etc. are sound cues.

Judge on, in this priority order:
1. Faithfulness: every claim is supported by the dialogue; no invented events, names or outside movie knowledge.
2. Coverage: captures the most important events / information of the segment.
3. Conciseness: 2-3 sentences, no filler, no repetition.
4. Clarity: fluent, unambiguous, third person.

Reply with ONLY a JSON object:
{"faithfulness_A": 1-5, "faithfulness_B": 1-5, "coverage_A": 1-5, "coverage_B": 1-5, "winner": "A" | "B" | "tie", "reason": "<15 words>"}"""


def judge_once(url, model, key, dialogue, a, b, timeout=180):
    import requests
    headers = {"Content-Type": "application/json"}
    if key:
        k = key.strip()
        headers["Authorization"] = k if k.lower().startswith("bearer ") else f"Bearer {k}"
    payload = {
        "model": model, "temperature": 0, "max_tokens": 200, "stream": False,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"Dialogue:\n{dialogue}\n\nSummary A:\n{a}\n\nSummary B:\n{b}"},
        ],
    }
    for attempt in range(4):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code in (408, 429) or r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            content = (r.json()["choices"][0]["message"].get("content") or "").strip()
            m = re.search(r"\{.*\}", content, re.DOTALL)
            res = json.loads(m.group(0)) if m else {}
            if str(res.get("winner", "")).upper() in ("A", "B", "TIE"):
                res["winner"] = str(res["winner"]).upper()
                return res
            raise ValueError(f"unparseable judge output: {content[:120]}")
        except Exception as e:
            if attempt == 3:
                print(f"    judge failed: {e}")
                return None
            time.sleep(3 * 2 ** attempt)


def judge_pair(url, model, key, dialogue, s1, s2):
    """Run both orders. Returns (winner in {'1','2','tie'}, faith1, faith2, cov1, cov2) or None."""
    r1 = judge_once(url, model, key, dialogue, s1, s2)    # s1 = A
    r2 = judge_once(url, model, key, dialogue, s2, s1)    # s1 = B
    if not r1 or not r2:
        return None
    w1 = {"A": "1", "B": "2", "TIE": "tie"}[r1["winner"]]
    w2 = {"A": "2", "B": "1", "TIE": "tie"}[r2["winner"]]
    winner = w1 if w1 == w2 else "tie"                    # inconsistent across orders -> tie

    def avg(x, y):
        try:
            return (float(x) + float(y)) / 2
        except (TypeError, ValueError):
            return None
    return (winner,
            avg(r1.get("faithfulness_A"), r2.get("faithfulness_B")),
            avg(r1.get("faithfulness_B"), r2.get("faithfulness_A")),
            avg(r1.get("coverage_A"), r2.get("coverage_B")),
            avg(r1.get("coverage_B"), r2.get("coverage_A")))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def fmt(x, pct=False, nd=3):
    if x is None:
        return "-"
    return f"{100 * x:.1f}%" if pct else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description="Compare teacher summaries (no ground truth).")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="name=folder pairs, e.g. gemma=./summaries_gemma qwen=./summaries_qwen")
    ap.add_argument("-o", "--out_dir", default="./compare")
    ap.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-base",
                    help="HF NLI cross-encoder; '' to skip NLI")
    ap.add_argument("--judge_model", default="", help="Judge model name on API ('' = skip judge)")
    ap.add_argument("--judge_url", default=os.getenv("JUDGE_API_URL", ""),
                    help="OpenAI-compatible /chat/completions URL for the judge")
    ap.add_argument("--judge_key_env", default="GEMMA_API_KEY", help="Env var holding the judge API key")
    ap.add_argument("--judge_sample", type=int, default=200, help="Chunks judged per model pair (0 = all)")
    ap.add_argument("--human_sample", type=int, default=50, help="Rows in the blind human-review sheet")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    runs = {}
    for spec in args.runs:
        name, _, folder = spec.partition("=")
        if not folder:
            ap.error(f"--runs expects name=folder, got '{spec}'")
        runs[name] = load_run(folder)
        print(f"{name}: {len(runs[name])} summaries from {folder}")
    names = list(runs)

    # Only chunks every model summarised, with the same input text
    common = set.intersection(*(set(r) for r in runs.values()))
    common = sorted(k for k in common
                    if len({runs[n][k]["text"] for n in names}) == 1)
    if not common:
        print("No common chunks across runs (same movie + interval + input text). Nothing to compare.")
        return
    print(f"Common chunks: {len(common)}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- per-chunk cheap metrics + NLI
    nli = None
    if args.nli_model:
        try:
            nli = NLIScorer(args.nli_model)
            print(f"NLI model loaded: {args.nli_model} on {nli.device}")
        except Exception as e:
            print(f"[WARN] NLI skipped ({type(e).__name__}: {e})")

    rows = []
    for idx, key in enumerate(common):
        text = runs[names[0]][key]["text"]
        row = {"movie": key[0], "interval": key[1], "input_words": len(text.split())}
        for n in names:
            s = runs[n][key]["summary"]
            nw, ns = len(s.split()), len(sentences(s))
            row[f"{n}_summary"] = s
            row[f"{n}_words"] = nw
            row[f"{n}_sentences"] = ns
            row[f"{n}_in_spec"] = int(SPEC_MIN_SENT <= ns <= SPEC_MAX_SENT and nw <= SPEC_MAX_WORDS)
            row[f"{n}_unknown_names"] = ", ".join(unknown_names(s, text))
            if nli:
                f_mean, f_frac = nli.score(text, s)
                row[f"{n}_nli_faith"] = round(f_mean, 4)
                row[f"{n}_nli_supported_sent"] = round(f_frac, 4)
        for a, b in itertools.combinations(names, 2):
            sa, sb = runs[a][key]["summary"], runs[b][key]["summary"]
            row[f"rouge1_{a}_vs_{b}"] = round(rouge1_f(sa, sb), 4)
            row[f"rougeL_{a}_vs_{b}"] = round(rougeL_f(sa, sb), 4)
        rows.append(row)
        if nli and (idx + 1) % 50 == 0:
            print(f"  NLI scored {idx + 1}/{len(common)}")

    # ---- LLM judge (pairwise, both orders)
    judge_stats = {}
    if args.judge_model and args.judge_url:
        key = os.getenv(args.judge_key_env, "")
        if args.judge_model in names:
            print(f"[WARN] judge '{args.judge_model}' is also a candidate: expect self-preference bias.")
        by_key = {(r["movie"], r["interval"]): r for r in rows}
        for a, b in itertools.combinations(names, 2):
            sample = common if args.judge_sample <= 0 else \
                random.sample(common, min(args.judge_sample, len(common)))
            wins = {"1": 0, "2": 0, "tie": 0}
            print(f"Judging {a} vs {b} on {len(sample)} chunks with {args.judge_model} ...")
            for i, k in enumerate(sample):
                res = judge_pair(args.judge_url, args.judge_model, key, runs[a][k]["text"],
                                 runs[a][k]["summary"], runs[b][k]["summary"])
                if res is None:
                    continue
                w, fa, fb, ca, cb = res
                wins[w] += 1
                r = by_key[k]
                r[f"judge_{a}_vs_{b}"] = {"1": a, "2": b, "tie": "tie"}[w]
                r[f"{a}_judge_faith"], r[f"{b}_judge_faith"] = fa, fb
                r[f"{a}_judge_cov"], r[f"{b}_judge_cov"] = ca, cb
                if (i + 1) % 25 == 0:
                    print(f"  {i + 1}/{len(sample)}  {a}:{wins['1']} {b}:{wins['2']} tie:{wins['tie']}")
            judge_stats[(a, b)] = wins
    elif args.judge_model:
        print("[WARN] --judge_model given without --judge_url; judge skipped.")

    # ---- write details CSV
    cols = list(dict.fromkeys(c for r in rows for c in r))
    with open(out / "compare_details.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---- blind human-review sheet (first two models)
    if len(names) >= 2:
        a, b = names[0], names[1]
        sample = random.sample(common, min(args.human_sample, len(common)))
        with open(out / "human_review.csv", "w", newline="", encoding="utf-8-sig") as f, \
             open(out / "human_review_key.csv", "w", newline="", encoding="utf-8-sig") as fk:
            w = csv.writer(f)
            wk = csv.writer(fk)
            w.writerow(["id", "movie", "interval", "dialogue", "summary_A", "summary_B",
                        "better (A/B/tie)", "A_hallucination (y/n)", "B_hallucination (y/n)", "notes"])
            wk.writerow(["id", "A", "B"])
            for i, k in enumerate(sample):
                pair = [a, b]
                random.shuffle(pair)
                w.writerow([i, k[0], k[1], runs[a][k]["text"],
                            runs[pair[0]][k]["summary"], runs[pair[1]][k]["summary"], "", "", "", ""])
                wk.writerow([i, pair[0], pair[1]])

    # ---- aggregate + report
    agg = {}
    for n in names:
        agg[n] = {
            "words": mean(r[f"{n}_words"] for r in rows),
            "sentences": mean(r[f"{n}_sentences"] for r in rows),
            "compression": mean(r[f"{n}_words"] / max(1, r["input_words"]) for r in rows),
            "in_spec": mean(r[f"{n}_in_spec"] for r in rows),
            "unknown_name_rate": mean(int(bool(r[f"{n}_unknown_names"])) for r in rows),
            "nli_faith": mean(r.get(f"{n}_nli_faith") for r in rows),
            "nli_supported": mean(r.get(f"{n}_nli_supported_sent") for r in rows),
            "judge_faith": mean(r.get(f"{n}_judge_faith") for r in rows),
            "judge_cov": mean(r.get(f"{n}_judge_cov") for r in rows),
        }

    L = [f"# Teacher summary comparison", "",
         f"Models: {', '.join(names)}  |  common chunks: {len(common)}", ""]
    L += ["| Metric | " + " | ".join(names) + " | Better |", "|---|" + "---|" * (len(names) + 1)]
    spec = [
        ("Avg words", "words", False, None),
        ("Avg sentences", "sentences", False, None),
        ("Compression (summary/input words)", "compression", True, None),
        (f"Within spec ({SPEC_MIN_SENT}-{SPEC_MAX_SENT} sent, <= {SPEC_MAX_WORDS} words)", "in_spec", True, "high"),
        ("Chunks with names not in dialogue", "unknown_name_rate", True, "low"),
        ("NLI faithfulness (mean entail prob)", "nli_faith", False, "high"),
        ("NLI supported sentences", "nli_supported", True, "high"),
        ("Judge faithfulness (1-5)", "judge_faith", False, "high"),
        ("Judge coverage (1-5)", "judge_cov", False, "high"),
    ]
    score = {n: 0 for n in names}
    for label, k, pct, better in spec:
        vals = {n: agg[n][k] for n in names}
        if all(v is None for v in vals.values()):
            continue
        best = ""
        valid = {n: v for n, v in vals.items() if v is not None}
        if better and len(valid) == len(names) and len(set(round(v, 4) for v in valid.values())) > 1:
            best = (max if better == "high" else min)(valid, key=valid.get)
            score[best] += 1
        L.append(f"| {label} | " + " | ".join(fmt(vals[n], pct) for n in names) + f" | {best} |")

    if judge_stats:
        L += ["", "## LLM judge (pairwise, both orders; inconsistent verdicts = tie)", ""]
        for (a, b), wv in judge_stats.items():
            tot = sum(wv.values()) or 1
            L.append(f"- **{a}** wins {wv['1']} ({100 * wv['1'] / tot:.1f}%), "
                     f"**{b}** wins {wv['2']} ({100 * wv['2'] / tot:.1f}%), tie {wv['tie']} "
                     f"({100 * wv['tie'] / tot:.1f}%)  - n={tot}")
            if wv["1"] != wv["2"]:
                score[a if wv["1"] > wv["2"] else b] += 2      # judge counts double

    for a, b in itertools.combinations(names, 2):
        L.append(f"\nAgreement {a} vs {b}: ROUGE-1 {fmt(mean(r[f'rouge1_{a}_vs_{b}'] for r in rows))}, "
                 f"ROUGE-L {fmt(mean(r[f'rougeL_{a}_vs_{b}'] for r in rows))} (similarity only, not quality)")

    ranked = sorted(score.items(), key=lambda x: -x[1])
    L += ["", "## Verdict", "",
          f"Points (1 per metric won, judge win-rate counts 2): "
          + ", ".join(f"{n}={s}" for n, s in ranked), ""]
    if len(ranked) > 1 and ranked[0][1] > ranked[1][1]:
        L.append(f"**Suggested teacher: {ranked[0][0]}.** Confirm with the blind human sheet "
                 f"(human_review.csv) before committing.")
    else:
        L.append("**No clear winner.** Use the human sheet, or pick per chunk "
                 "(higher NLI faithfulness, then judge preference).")
    L += ["", "Notes: NLI is truncated to 512 tokens per window (sliding windows over the dialogue). "
              "Unknown-name check is heuristic (capitalised words absent from the dialogue)."]

    report = "\n".join(L)
    (out / "compare_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nFiles written to {out.resolve()}")


if __name__ == "__main__":
    main()
