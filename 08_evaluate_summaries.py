#!/usr/bin/env python3
"""
08_evaluate_summaries.py

Full evaluation report for subtitle-summarization models (for reporting).

Compares any number of SYSTEMS (e.g. T5 fp32, T5 INT8, SmolLM, baseline vs filtered)
against the TEACHER summaries and against the SOURCE text, and adds REFERENCE
POINTS so the numbers can be interpreted:

  * lead_baseline     : first words of the dialogue, cut to the teacher's average
                        length (a trivial extractive baseline; floor)
  * second teacher    : e.g. Qwen summaries scored against Gemma (--second_teacher).
                        Shows how much two strong LLMs agree with each other; a student
                        close to this is close to the practical ceiling.
  * teacher vs source : grounding scores of the teacher itself (ceiling for the
                        source-based metrics).

Metrics
-------
vs teacher (reference):  ROUGE-1/2/L F1, BLEU, chrF, BERTScore P/R/F1 (raw + rescaled),
                         cosine similarity (sentence embeddings)
vs source dialogue:      NLI faithfulness, cosine to source, novel-name rate,
                         [optional] LLM-judge faithful % (different model, e.g. Qwen)
output quality:          avg words, length ratio vs teacher, % in 2-3 sentences,
                         % repeated trigrams, % empty
95% bootstrap confidence intervals for the main metrics.

Inputs
------
System files: jsonl/json (or a folder of them) from 07_finetune_t5.py predict, i.e.
records with "input" (or "preprocessed_input") and "summary" (or "student"/"prediction"),
optionally "teacher". Records are matched across files by their input text.
--reference : teacher summaries (folder of <movie>_output.json, or jsonl). If omitted,
              the "teacher" field inside the first system file is used.

Example
-------
python 08_evaluate_summaries.py \
    --system t5_fp32=preds_fp32.jsonl --system t5_int8=preds_int8.jsonl \
    --reference ./summaries_gemma --second_teacher ./summaries_qwen \
    -o ./eval_report

Optional packages (each metric is skipped with a warning if missing):
    pip install rouge-score bert-score sentence-transformers sacrebleu transformers torch
"""

import argparse
import csv
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

SENT_RE = re.compile(r"(?<=[.!?])\s+")
NOT_NAMES = set("""
I A An The He She They It We You His Her Their Its Our Your Him Them This That These Those There Here Then
When While After Before As At In On Of For With Without From To Into Meanwhile However Later Finally Also
Both Two Three One Another Some Several Someone Somebody Everyone Nobody Each Others
Mr Mrs Ms Miss Dr Sir Madam Mom Mum Dad Mother Father Grandma Grandpa God Lord Okay OK Yes No
Monday Tuesday Wednesday Thursday Friday Saturday Sunday Christmas English French German Spanish
Tension Things Everything Nothing Something Life Love Music Conversation Discussion Talk Danger Fear Panic
Chaos Trouble News Time Night Morning Evening Police Officer Doctor Captain Boss Agent Detective
""".split())
_NOUN_SUFFIX = ("ion", "ness", "ment", "ity", "ence", "ance", "ship", "ing", "ism", "hood", "ure", "ics")
_STARTER_VERBS = set("""asks tells says argues discusses explains reveals warns admits insists refuses
agrees accuses confronts meets talks calls tries wants needs plans decides learns finds realizes
describes questions urges begs offers threatens apologizes complains jokes""".split())


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _norm(t):
    return re.sub(r"\s+", " ", t or "").strip().lower()


def _iter_records(path):
    p = Path(path)
    files = [p] if p.is_file() else sorted(x for x in p.rglob("*")
                                           if x.suffix.lower() in (".json", ".jsonl")
                                           and not x.name.endswith("_chunk.json"))
    for fp in files:
        try:
            if fp.suffix.lower() == ".jsonl":
                recs = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
            else:
                obj = json.load(open(fp, encoding="utf-8"))
                recs = obj if isinstance(obj, list) else next(
                    (obj[k] for k in ("results", "data", "summaries") if isinstance(obj.get(k), list)), [])
        except Exception as e:
            print(f"  [skip] {fp}: {e}")
            continue
        movie = re.sub(r"_(output|summary|summaries|preds?)$", "", fp.stem)
        for r in recs:
            if isinstance(r, dict):
                yield movie, r


def load_system(path):
    """{norm_input: {"input", "summary", "teacher", "movie"}}"""
    out = {}
    for movie, r in _iter_records(path):
        src = r.get("input") or r.get("preprocessed_input") or r.get("source_text") or ""
        summ = next((r[k] for k in ("summary", "student", "prediction", "pred", "generated_summary",
                                    "reference_summary") if isinstance(r.get(k), str)), "")
        if not src.strip() or r.get("status", "ok") != "ok" or summ.upper().startswith("ERROR"):
            continue
        out[_norm(src)] = {"input": src, "summary": summ.strip(), "teacher": (r.get("teacher") or r.get("target") or "").strip(),
                           "movie": r.get("movie") or movie}
    return out


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def sentences(t):
    return [s for s in SENT_RE.split(t.strip()) if s.strip()]


def toks(t):
    return re.findall(r"[a-z0-9']+", t.lower())


def novel_names(summary, source):
    src = {re.sub(r"'s?$", "", w) for w in re.findall(r"[a-z][a-z'\-]*", source.lower())}
    found = []
    for sent in sentences(summary):
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", sent)
        for i, w in enumerate(words):
            if not w[:1].isupper():
                continue
            if i == 0:
                nxt = words[1] if len(words) > 1 else ""
                if not (w.endswith("'s") or nxt.lower() in ("and", "or", "with") or nxt[:1].isupper()
                        or nxt.lower() in _STARTER_VERBS
                        or (nxt.endswith("s") and nxt.islower() and not w.lower().endswith(_NOUN_SUFFIX))):
                    continue
            base = re.sub(r"'s?$", "", w)
            if len(base) >= 2 and base not in NOT_NAMES and base.lower() not in src and base not in found:
                found.append(base)
    return found


def repeated_trigram_rate(t):
    tk = toks(t)
    g = [tuple(tk[i:i + 3]) for i in range(len(tk) - 2)]
    return 0.0 if not g else 1 - len(set(g)) / len(g)


def lead_baseline(source, n_words):
    """First n words of the dialogue (speaker labels / cue tags removed)."""
    t = re.sub(r"\[[a-z]+\]|\([a-z]+\)", " ", source)
    t = re.sub(r"\b[A-Z][\w .'-]{0,25}:\s", " ", t)
    w = t.split()
    return " ".join(w[:max(5, n_words)])


# --------------------------------------------------------------------------
# Metric back-ends (each optional)
# --------------------------------------------------------------------------
def rouge_all(preds, refs):
    try:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        res = [sc.score(r, p) for p, r in zip(preds, refs)]
        return {k: [x[k].fmeasure for x in res] for k in ("rouge1", "rouge2", "rougeL")}, "rouge-score (stemmed)"
    except ImportError:
        pass

    def ngf(a, b, n):
        ga = [tuple(a[i:i + n]) for i in range(len(a) - n + 1)]
        gb = [tuple(b[i:i + n]) for i in range(len(b) - n + 1)]
        if not ga or not gb:
            return 0.0
        cb = {}
        for x in gb:
            cb[x] = cb.get(x, 0) + 1
        ov = 0
        for x in ga:
            if cb.get(x, 0):
                cb[x] -= 1
                ov += 1
        return 0.0 if not ov else 2 * (ov / len(ga)) * (ov / len(gb)) / (ov / len(ga) + ov / len(gb))

    def lcsf(a, b):
        if not a or not b:
            return 0.0
        prev = [0] * (len(b) + 1)
        for x in a:
            cur = [0]
            for j, y in enumerate(b):
                cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
            prev = cur
        l = prev[-1]
        return 0.0 if not l else 2 * (l / len(a)) * (l / len(b)) / (l / len(a) + l / len(b))
    r1, r2, rl = [], [], []
    for p, r in zip(preds, refs):
        a, b = toks(p), toks(r)
        r1.append(ngf(a, b, 1)); r2.append(ngf(a, b, 2)); rl.append(lcsf(a, b))
    return {"rouge1": r1, "rouge2": r2, "rougeL": rl}, "built-in (no stemming; pip install rouge-score)"


def bleu_chrf(preds, refs):
    try:
        import sacrebleu
        return (sacrebleu.corpus_bleu(preds, [refs]).score / 100,
                sacrebleu.corpus_chrf(preds, [refs]).score / 100)
    except ImportError:
        return None, None


class BertScorer:
    def __init__(self, model_type, batch_size):
        import bert_score  # noqa: F401
        self.model_type, self.bs = model_type, batch_size

    def score(self, preds, refs):
        import bert_score
        P, R, F = bert_score.score(preds, refs, model_type=self.model_type, lang="en",
                                   batch_size=self.bs, verbose=False)
        out = {"bertscore_p_raw": P.tolist(), "bertscore_r_raw": R.tolist(), "bertscore_f1_raw": F.tolist()}
        try:
            P2, R2, F2 = bert_score.score(preds, refs, model_type=self.model_type, lang="en",
                                          batch_size=self.bs, rescale_with_baseline=True, verbose=False)
            out.update({"bertscore_p": P2.tolist(), "bertscore_r": R2.tolist(), "bertscore_f1": F2.tolist()})
        except Exception as e:
            print(f"  [warn] BERTScore rescaling unavailable ({e}); reporting raw only")
        return out


class Embedder:
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)

    def cos(self, a_list, b_list):
        ea = self.m.encode(a_list, convert_to_tensor=True, normalize_embeddings=True, batch_size=64)
        eb = self.m.encode(b_list, convert_to_tensor=True, normalize_embeddings=True, batch_size=64)
        return (ea * eb).sum(-1).tolist()


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
        sents = sentences(summary) or [summary or "."]
        pairs = [(x, s) for s in sents for x in wins]
        probs = []
        for i in range(0, len(pairs), 16):
            b = pairs[i:i + 16]
            enc = self.tok([p for p, _ in b], [h for _, h in b], truncation="only_first",
                           max_length=512, padding=True, return_tensors="pt").to(self.dev)
            with self.torch.no_grad():
                probs += self.m(**enc).logits.softmax(-1)[:, self.ent].tolist()
        per = [max(probs[k * len(wins):(k + 1) * len(wins)]) for k in range(len(sents))]
        return sum(per) / len(per)


JUDGE_SYSTEM = """You check whether a summary of movie subtitle dialogue is faithful to the dialogue.
"Name:" at the start of a sentence is a speaker label; [music], [laughter] etc. are sound cues.
A claim is SUPPORTED if stated in the dialogue OR a reasonable inference from it
(e.g. "a mother and her son" when one says "Mom"; "a film crew shoots a scene" after "Action!").
A claim is UNSUPPORTED if it adds names, places, events, outcomes, motives or relationships the
dialogue does not state or clearly imply, or contradicts it.
Reply with ONLY: {"faithful": "yes" or "no"}"""


def judge(url, model, key, text, summary):
    import requests
    h = {"Content-Type": "application/json"}
    if key:
        k = key.strip()
        h["Authorization"] = k if k.lower().startswith("bearer ") else f"Bearer {k}"
    payload = {"model": model, "temperature": 0, "max_tokens": 60, "stream": False,
               "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user", "content": f"Dialogue:\n{text}\n\nSummary:\n{summary}"}]}
    for a in range(4):
        try:
            r = requests.post(url, headers=h, json=payload, timeout=180)
            r.raise_for_status()
            c = r.json()["choices"][0]["message"].get("content") or ""
            m = re.search(r'"faithful"\s*:\s*"(yes|no)"', c, re.I)
            if m:
                return m.group(1).lower() == "yes"
        except Exception:
            pass
        time.sleep(3 * 2 ** a)
    return None


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------
def mean_ci(vals, n_boot=1000, seed=0):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, None
    m = sum(v) / len(v)
    if len(v) < 5:
        return m, None, None
    rnd = random.Random(seed)
    boots = sorted(sum(rnd.choice(v) for _ in v) / len(v) for _ in range(n_boot))
    return m, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def fmt(m, lo=None, hi=None, pct=False, nd=3):
    if m is None:
        return "-"
    f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.{nd}f}")
    return f(m) + (f" [{f(lo)}–{f(hi)}]" if lo is not None else "")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Full evaluation report for summarization systems.")
    ap.add_argument("--system", action="append", required=True, help="name=path (repeatable)")
    ap.add_argument("--reference", default="", help="Teacher summaries (folder/json/jsonl). Default: 'teacher' field")
    ap.add_argument("--second_teacher", default="", help="Another LLM's summaries, e.g. Qwen (agreement reference)")
    ap.add_argument("-o", "--out_dir", default="./eval_report")
    ap.add_argument("--no_lead_baseline", action="store_true")
    ap.add_argument("--bertscore_model", default="roberta-large", help="'' to skip BERTScore")
    ap.add_argument("--embed_model", default="sentence-transformers/all-mpnet-base-v2", help="'' to skip cosine")
    ap.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-base", help="'' to skip NLI")
    ap.add_argument("--judge_model", default="", help="LLM judge (different model than the teacher), '' = skip")
    ap.add_argument("--judge_url", default=os.getenv("JUDGE_API_URL", ""))
    ap.add_argument("--judge_key_env", default="GEMMA_API_KEY")
    ap.add_argument("--judge_sample", type=int, default=200, help="Chunks per system for the judge (0 = all)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---- load
    systems = {}
    for spec in args.system:
        name, _, path = spec.partition("=")
        if not path:
            ap.error(f"--system expects name=path, got {spec}")
        systems[name] = load_system(path)
        print(f"{name}: {len(systems[name])} records")
    if args.reference:
        ref = {k: v["summary"] for k, v in load_system(args.reference).items()}
        print(f"reference (teacher): {len(ref)} summaries")
    else:
        first = next(iter(systems.values()))
        ref = {k: v["teacher"] for k, v in first.items() if v["teacher"]}
        print(f"reference: 'teacher' field of first system ({len(ref)})")
    if args.second_teacher:
        systems["second_teacher (agreement)"] = load_system(args.second_teacher)
        print(f"second teacher: {len(systems['second_teacher (agreement)'])} summaries")

    common = set(ref)
    for s in systems.values():
        common &= set(s)
    common = sorted(common)
    if not common:
        print("No chunk is present in every system AND the reference (matched by input text). "
              "Check that all files come from the same test chunks.")
        return
    print(f"Evaluating on {len(common)} common chunks")

    src = {k: next(iter(systems.values()))[k]["input"] for k in common}
    refs = [ref[k] for k in common]
    sources = [src[k] for k in common]
    preds = {n: [s[k]["summary"] for k in common] for n, s in systems.items()}
    avg_ref_words = st.mean(len(r.split()) for r in refs)
    if not args.no_lead_baseline:
        preds["lead_baseline"] = [lead_baseline(s, round(avg_ref_words)) for s in sources]
    preds["teacher (vs source only)"] = refs
    names = list(preds)

    # ---- load models once
    def try_load(fn, label):
        try:
            return fn()
        except Exception as e:
            print(f"[warn] {label} skipped: {type(e).__name__}: {e}")
            return None
    bs = try_load(lambda: BertScorer(args.bertscore_model, args.batch_size), "BERTScore") if args.bertscore_model else None
    emb = try_load(lambda: Embedder(args.embed_model), "cosine similarity") if args.embed_model else None
    nli = try_load(lambda: NLI(args.nli_model), "NLI") if args.nli_model else None

    per = {}      # name -> metric -> list
    corpus = {}   # name -> corpus-level metrics
    rouge_impl = ""
    for n in names:
        p = preds[n]
        is_teacher = n.startswith("teacher")
        m = {}
        print(f"\nScoring {n} ...")
        if not is_teacher:
            r, rouge_impl = rouge_all(p, refs)
            m.update(r)
            b, c = bleu_chrf(p, refs)
            corpus[n] = {"bleu": b, "chrf": c}
            if bs:
                m.update(bs.score(p, refs))
            if emb:
                m["cosine_to_teacher"] = emb.cos(p, refs)
            m["length_ratio"] = [len(x.split()) / max(1, len(r.split())) for x, r in zip(p, refs)]
        if emb:
            m["cosine_to_source"] = emb.cos(p, sources)
        if nli:
            m["nli_faithfulness"] = [nli.score(s, x) for s, x in zip(sources, p)]
        m["novel_names"] = [int(bool(novel_names(x, s))) for x, s in zip(p, sources)]
        m["words"] = [len(x.split()) for x in p]
        m["in_2_3_sentences"] = [int(2 <= len(sentences(x)) <= 3) for x in p]
        m["repeated_trigrams"] = [repeated_trigram_rate(x) for x in p]
        m["empty"] = [int(not x.strip()) for x in p]
        per[n] = m

    # ---- optional LLM judge
    if args.judge_model and args.judge_url:
        key = os.getenv(args.judge_key_env, "")
        rnd = random.Random(args.seed)
        idx = list(range(len(common)))
        sample = idx if args.judge_sample <= 0 else rnd.sample(idx, min(args.judge_sample, len(idx)))
        for n in names:
            if n == "lead_baseline":
                continue
            print(f"Judging {n} on {len(sample)} chunks ...")
            vals = [None] * len(common)
            for j, i in enumerate(sample):
                v = judge(args.judge_url, args.judge_model, key, sources[i], preds[n][i])
                vals[i] = None if v is None else int(v)
                if (j + 1) % 50 == 0:
                    print(f"  {j + 1}/{len(sample)}")
            per[n]["judge_faithful"] = vals

    # ---- write per-sample CSV
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for m in per.values() for k in m})
    with open(out / "per_sample.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["system", "movie", "input", "teacher", "summary"] + keys)
        for n in names:
            for i, k in enumerate(common):
                mv = next(iter(systems.values()))[k]["movie"]
                w.writerow([n, mv, sources[i], refs[i], preds[n][i]] +
                           [per[n].get(x, [None] * len(common))[i] for x in keys])

    # ---- report
    rows_ref = [
        ("rouge1", "ROUGE-1 F1", False), ("rouge2", "ROUGE-2 F1", False), ("rougeL", "ROUGE-L F1", False),
        ("bleu", "BLEU (corpus)", False), ("chrf", "chrF (corpus)", False),
        ("bertscore_f1", "BERTScore F1 (rescaled)", False), ("bertscore_p", "BERTScore P (rescaled)", False),
        ("bertscore_r", "BERTScore R (rescaled)", False), ("bertscore_f1_raw", "BERTScore F1 (raw)", False),
        ("cosine_to_teacher", "Cosine sim. to teacher", False),
    ]
    rows_src = [
        ("nli_faithfulness", "NLI faithfulness (strict)", False),
        ("judge_faithful", "LLM-judge faithful", True),
        ("cosine_to_source", "Cosine sim. to source", False),
        ("novel_names", "Summaries with invented names", True),
    ]
    rows_out = [
        ("words", "Avg words", False), ("length_ratio", "Length ratio vs teacher", False),
        ("in_2_3_sentences", "In 2-3 sentences", True), ("repeated_trigrams", "Repeated trigrams", True),
        ("empty", "Empty outputs", True),
    ]
    metrics_json = {n: {} for n in names}

    def table(title, spec, include):
        cols = [n for n in names if include(n)]
        L = [f"### {title}", "", "| Metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for k, label, pct in spec:
            cells, any_val = [], False
            for n in cols:
                if k in ("bleu", "chrf"):
                    v = (corpus.get(n) or {}).get(k)
                    cells.append(fmt(v, pct=pct)); any_val |= v is not None
                    metrics_json[n][k] = v
                else:
                    m, lo, hi = mean_ci(per[n].get(k, []))
                    nd = 1 if k == "words" else 3
                    cells.append(fmt(m, lo, hi, pct=pct, nd=nd)); any_val |= m is not None
                    metrics_json[n][k] = {"mean": m, "ci95": [lo, hi]}
            if any_val:
                L.append(f"| {label} | " + " | ".join(cells) + " |")
        return L + [""]

    L = ["# Subtitle summarization — evaluation report", "",
         f"Chunks evaluated: **{len(common)}** (identical inputs for every system). "
         f"Values are means with 95% bootstrap confidence intervals.", ""]
    L += table("1. Agreement with teacher summaries (meaning and wording)", rows_ref,
               lambda n: not n.startswith("teacher"))
    L += table("2. Grounding in the source dialogue (hallucination checks)", rows_src, lambda n: True)
    L += table("3. Output quality", rows_out, lambda n: True)
    L += ["### How to read this", "",
          "- **lead_baseline** = first words of the dialogue; a trivial floor. Any useful model must beat it on "
          "meaning (BERTScore / cosine) and grounding.",
          "- **second_teacher (agreement)** (if present) = how closely a second strong LLM matches the teacher. "
          "Valid summaries differ in wording, so this is a reference point: a student near or above it matches "
          "the teacher about as closely as another large LLM does (the student is trained to imitate the "
          "teacher, so it can exceed this).",
          "- **teacher (vs source only)** = the teacher's own grounding; the ceiling for section 2.",
          "- ROUGE / BLEU count exact word overlap and penalise valid paraphrases; BERTScore and cosine measure "
          "meaning. Rescaled BERTScore spreads the raw 0.85–0.90 range so differences are visible.",
          "- NLI faithfulness is strict (dialogue → narrative inferences score low); the LLM-judge % is the more "
          "realistic faithfulness figure.", "",
          f"ROUGE implementation: {rouge_impl}. BERTScore model: {args.bertscore_model or 'skipped'}. "
          f"Embeddings: {args.embed_model or 'skipped'}. NLI: {args.nli_model or 'skipped'}. "
          f"Judge: {args.judge_model or 'skipped'}."]
    report = "\n".join(L)
    (out / "report.md").write_text(report, encoding="utf-8")
    json.dump(metrics_json, open(out / "metrics.json", "w"), indent=2)
    print("\n" + report + f"\n\nFiles: {out.resolve()}  (report.md, metrics.json, per_sample.csv)")


if __name__ == "__main__":
    main()
