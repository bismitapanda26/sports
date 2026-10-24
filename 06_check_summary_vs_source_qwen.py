#!/usr/bin/env python3
"""
06_check_summary_vs_source.py

Check ONE model's summaries against their own preprocessed text (no ground truth,
no second model). Use it to measure label quality and find bad summaries.

Input : folder with <movie>_output.json (interval / preprocessed_input / summary)
Output: <out_dir>/summary_check.csv  - per-chunk scores + flags, worst first
        <out_dir>/summary_check.md   - averages, flag counts, 10 worst chunks

Per chunk
---------
faithfulness   NLI: is each summary sentence supported by the text? (0-1, higher better)  [main]
semantic_sim   embedding similarity summary vs text (0-1, higher better)
rouge1_p / rouge2_p / rougeL_p
               share of summary words / bigrams / sequence found in the text
novel_names    names in the summary that are NOT in the text (should be empty)
words, sentences, compression

LLM judge (optional, recommended)
--------------------------------
A DIFFERENT model (e.g. Qwen) reads the text + summary and answers: is every claim
supported by the text (reasonable inferences allowed)? yes/no + the unsupported claim.
NLI cannot make inferences ("Action! ... [laughter]" -> filming bloopers), the judge can.
--judge_scope low  : judge only chunks with NLI faithfulness < 0.5 (default, cheaper)
--judge_scope all  : judge every chunk
Judge answers are cached in <out_dir>/judge_cache.jsonl (re-runs do not re-call the API).

Flags
-----
novel_names, judge_unfaithful, too_long (>50 words), bad_sentence_count (not 2-3)  -> DROP
low_nli (<0.5), off_topic (sim < 0.3)                                              -> info only

Filtered training data (only kept chunks) is written to <out_dir>/filtered/<movie>_output.json
with interval / preprocessed_input / summary.

Example
-------
python 06_check_summary_vs_source.py -i ./summaries_gemma -o ./check_gemma \
    --judge_model <qwen-model-name> --judge_url http://<server>:4000/v1/chat/completions
"""

import argparse
import csv
import hashlib
import json
import os
import re
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pathlib import Path

SENT_RE = re.compile(r"(?<=[.!?])\s+")
NOT_NAMES = set("""
I A An The He She They It We You His Her Their Its Our Your This That These Those There Here Then
When While After Before As At In On Of For With Without From To Into Meanwhile However Later Finally
Mr Mrs Ms Miss Dr Sir Madam Mom Mum Dad Mother Father Grandma Grandpa God Lord Okay OK Yes No
Him Them Also Both Two Three One Another Some Several Someone Somebody Everyone Nobody Each Others
Monday Tuesday Wednesday Thursday Friday Saturday Sunday Christmas English French German Spanish
Tension Things Everything Nothing Something Life Love Music Conversation Discussion Talk Danger Fear Panic
Chaos Trouble News Time Night Morning Evening Police Officer Doctor Captain Boss Agent Detective
""".split())


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
SUMMARY_KEYS = ("summary", "reference_summary", "generated_summary", "pred_summary")
SOURCE_KEYS = ("preprocessed_input", "source_text", "text", "input")
LABEL_KEYS = ("interval", "window_label")


def _norm(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def _records(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("results", "data", "summaries", "chunks"):
            if isinstance(obj.get(k), list):
                return obj[k]
    return []


def load(folder, name):
    """
    Accepts old and new generator output (summary / reference_summary,
    interval / window_label, preprocessed_input / source_text), .json or .jsonl,
    in sub-folders too. Chunks are matched across models by their SOURCE TEXT,
    so file names and interval formats do not need to match.
    """
    folder = Path(folder)
    if not folder.exists():
        print(f"[ERROR] {name}: folder '{folder}' does not exist")
        return {}
    files = sorted(p for p in folder.rglob("*")
                   if p.suffix.lower() in (".json", ".jsonl") and not p.name.endswith("_chunk.json"))
    data, n_rec, n_err, n_nosum, keys_seen = {}, 0, 0, 0, set()
    for fp in files:
        try:
            if fp.suffix.lower() == ".jsonl":
                recs = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
            else:
                recs = _records(json.load(open(fp, encoding="utf-8")))
        except Exception as e:
            print(f"  [skip] {fp.name}: {e}")
            continue
        movie = re.sub(r"_(output|summary|summaries)$", "", fp.stem)
        for r in recs:
            if not isinstance(r, dict):
                continue
            n_rec += 1
            keys_seen.update(r.keys())
            s = next((r[k] for k in SUMMARY_KEYS if isinstance(r.get(k), str) and r[k].strip()), "").strip()
            src = next((r[k] for k in SOURCE_KEYS if isinstance(r.get(k), str) and r[k].strip()), "")
            if not s or not src:
                n_nosum += 1
                continue
            if s.upper().startswith("ERROR") or r.get("status", "ok") != "ok":
                n_err += 1
                continue
            label = next((r[k] for k in LABEL_KEYS if r.get(k)), "")
            data[_norm(src)] = (movie, label, src, s)
    print(f"{name}: {len(files)} files, {n_rec} records -> {len(data)} usable "
          f"({n_err} error/failed, {n_nosum} missing summary or source)")
    if n_rec and not data:
        print(f"  keys found in records: {sorted(keys_seen)}")
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


_NOUN_SUFFIX = ("ion", "ness", "ment", "ity", "ence", "ance", "ship", "ing", "ism", "hood", "ure", "ics")
_STARTER_VERBS = set("""asks tells says argues discusses explains reveals warns admits insists refuses
agrees accuses confronts meets talks calls tries wants needs plans decides learns finds realizes
describes questions urges begs offers threatens apologizes complains jokes""".split())


def novel_names(summary, source):
    """Same check as the generator: proper nouns not present as a whole word in the dialogue."""
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
# LLM judge
# --------------------------------------------------------------------------
JUDGE_SYSTEM = """You check whether a summary of movie subtitle dialogue is faithful to the dialogue.
"Name:" at the start of a sentence is a speaker label; [music], [laughter], [crying] etc. are sound cues.

A claim is SUPPORTED if it is stated in the dialogue OR is a reasonable inference from it
(e.g. "a mother and her son" when one says "Mom"; "a film crew shoots a scene" when people say
"Action!" and "keep rolling"). Paraphrasing and summarising are fine.
A claim is UNSUPPORTED if it adds names, places, events, outcomes, motives or relationships
that the dialogue does not state or clearly imply, or if it contradicts the dialogue.

Reply with ONLY a JSON object:
{"faithful": "yes" or "no", "unsupported": "<the unsupported claim, or empty>", "reason": "<max 15 words>"}"""


_NO_THINK_FLAG = {"supported": True}     # set False once the server rejects chat_template_kwargs


def _judge_call(url, model, key, text, summary, timeout=180):
    import requests
    headers = {"Content-Type": "application/json"}
    if key:
        k = key.strip()
        if k.lower().startswith("bearer "):
            k = k[7:].strip()
        headers["Authorization"] = f"Bearer {k}"
    payload = {"model": model, "temperature": 0, "max_tokens": 200, "stream": False,
               "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user", "content": f"Dialogue:\n{text}\n\nSummary:\n{summary}"}]}
    if _NO_THINK_FLAG["supported"]:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    last = None
    for attempt in range(5):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 400 and "chat_template_kwargs" in payload:
                payload.pop("chat_template_kwargs")          # server does not support it
                _NO_THINK_FLAG["supported"] = False
                continue
            if r.status_code in (408, 429) or r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            content = (r.json()["choices"][0]["message"].get("content") or "")
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
            m = re.search(r"\{.*\}", content, re.S)
            res = json.loads(m.group(0)) if m else {}
            verdict = str(res.get("faithful", "")).strip().lower()
            if verdict in ("yes", "no"):
                return {"faithful": verdict, "unsupported": str(res.get("unsupported", "")),
                        "reason": str(res.get("reason", ""))}
            raise ValueError(f"unparseable: {content[:100]}")
        except Exception as e:
            last = e
            time.sleep(3 * 2 ** attempt)
    return {"faithful": "error", "unsupported": "", "reason": str(last)[:150]}


class Judge:
    def __init__(self, url, model, key, cache_path):
        self.url, self.model, self.key, self.cache_path = url, model, key, cache_path
        self.cache = {}
        if cache_path.exists():
            for line in open(cache_path, encoding="utf-8"):
                try:
                    d = json.loads(line)
                    self.cache[d["key"]] = d["result"]
                except Exception:
                    pass

    def _key(self, text, summary):
        return hashlib.sha1(f"{self.model}\n{text}\n{summary}".encode("utf-8")).hexdigest()

    def run(self, items, workers=4):
        """items: list of (text, summary). Returns list of result dicts."""
        todo = [(i, t, s) for i, (t, s) in enumerate(items) if self._key(t, s) not in self.cache]
        print(f"Judge {self.model}: {len(items)} chunks ({len(items) - len(todo)} cached, {len(todo)} to call)")
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex, \
                open(self.cache_path, "a", encoding="utf-8") as f:
            futs = {ex.submit(_judge_call, self.url, self.model, self.key, t, s): (i, t, s)
                    for i, t, s in todo}
            for fut in futs:
                i, t, s = futs[fut]
                res = fut.result()
                k = self._key(t, s)
                if res["faithful"] != "error":
                    f.write(json.dumps({"key": k, "result": res}, ensure_ascii=False) + "\n")
                    f.flush()
                self.cache[k] = res
                done += 1
                if done % 25 == 0:
                    print(f"  judged {done}/{len(todo)}")
        return [self.cache[self._key(t, s)] for t, s in items]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
FAITH_MIN, SIM_MIN, MAX_WORDS, SENT_RANGE = 0.5, 0.3, 50, (2, 3)
DROP_FLAGS = ("novel_names", "judge_unfaithful", "too_long", "bad_sentence_count")


def main():
    ap = argparse.ArgumentParser(description="Check summaries against their preprocessed text.")
    ap.add_argument("-i", "--input", required=True, help="Folder (or file) with *_output.json")
    ap.add_argument("-o", "--out_dir", default="./summary_check")
    ap.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-base", help="'' to skip")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2", help="'' to skip")
    ap.add_argument("--judge_model", default="", help="Judge model, e.g. Qwen ('' = no judge)")
    ap.add_argument("--judge_url", default=os.getenv("JUDGE_API_URL", ""),
                    help="OpenAI-compatible /chat/completions URL")
    ap.add_argument("--judge_key_env", default="GEMMA_API_KEY", help="Env var with the judge API key")
    ap.add_argument("--judge_scope", choices=["low", "all"], default="low",
                    help="low: only chunks with NLI < 0.5 (or all if NLI unavailable); all: every chunk")
    ap.add_argument("--workers", type=int, default=4, help="Parallel judge requests")
    args = ap.parse_args()

    inp = Path(args.input)
    data = load(inp if inp.is_dir() else inp.parent, inp.name)
    if inp.is_file():
        data = {k: v for k, v in data.items() if v[0] == re.sub(r"_(output|summary|summaries)$", "", inp.stem)}
    if not data:
        print("No summaries loaded. Check the folder path.")
        return
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

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

    # ---- automatic metrics
    rows = []
    for i, (movie, label, src, s) in enumerate(sorted(data.values(), key=lambda v: (v[0], v[1]))):
        names = novel_names(s, src)
        nw, ns = len(s.split()), len(sentences(s))
        rows.append({
            "movie": movie, "interval": label,
            "faithfulness": round(nli.score(src, s), 4) if nli else None,
            "semantic_sim": round(emb.sim(s, src), 4) if emb else None,
            "rouge1_p": round(ngram_precision(s, src, 1), 4),
            "rouge2_p": round(ngram_precision(s, src, 2), 4),
            "rougeL_p": round(lcs_precision(s, src), 4),
            "novel_names": ", ".join(names),
            "words": nw, "sentences": ns,
            "compression": round(nw / max(1, len(src.split())), 4),
            "judge_faithful": "", "judge_unsupported": "", "judge_reason": "",
            "summary": s, "preprocessed_input": src,
        })
        if (i + 1) % 50 == 0:
            print(f"  scored {i + 1}/{len(data)}")

    # ---- LLM judge
    judged = 0
    if args.judge_model:
        if not args.judge_url:
            print("[WARN] --judge_model given without --judge_url: judge skipped.")
        else:
            sel = [r for r in rows if args.judge_scope == "all" or r["faithfulness"] is None
                   or r["faithfulness"] < FAITH_MIN]
            judge = Judge(args.judge_url, args.judge_model, os.getenv(args.judge_key_env, ""),
                          out / "judge_cache.jsonl")
            results = judge.run([(r["preprocessed_input"], r["summary"]) for r in sel], args.workers)
            for r, res in zip(sel, results):
                r["judge_faithful"], r["judge_unsupported"], r["judge_reason"] = \
                    res["faithful"], res["unsupported"], res["reason"]
            judged = len(sel)

    # ---- flags / keep
    for r in rows:
        flags = []
        if r["novel_names"]:
            flags.append("novel_names")
        if r["judge_faithful"] == "no":
            flags.append("judge_unfaithful")
        if r["words"] > MAX_WORDS:
            flags.append("too_long")
        if not SENT_RANGE[0] <= r["sentences"] <= SENT_RANGE[1]:
            flags.append("bad_sentence_count")
        if r["faithfulness"] is not None and r["faithfulness"] < FAITH_MIN:
            flags.append("low_nli")
        if r["semantic_sim"] is not None and r["semantic_sim"] < SIM_MIN:
            flags.append("off_topic")
        r["flags"] = ";".join(flags)
        r["keep"] = "no" if any(f in DROP_FLAGS for f in flags) else "yes"

    # worst first: dropped first, then most flags, then lowest NLI
    rows.sort(key=lambda r: (r["keep"] == "yes", -len([x for x in r["flags"].split(";") if x]),
                             r["faithfulness"] if r["faithfulness"] is not None else r["rougeL_p"]))
    cols = ["movie", "interval", "keep", "flags", "judge_faithful", "judge_unsupported", "judge_reason",
            "faithfulness", "semantic_sim", "rouge1_p", "rouge2_p", "rougeL_p", "novel_names",
            "words", "sentences", "compression", "summary", "preprocessed_input"]
    with open(out / "summary_check.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---- filtered training data
    fdir = out / "filtered"
    fdir.mkdir(exist_ok=True)
    by_movie = {}
    for r in rows:
        if r["keep"] == "yes":
            by_movie.setdefault(r["movie"], []).append(r)
    for movie, rs in by_movie.items():
        rs.sort(key=lambda r: data_order(r["interval"]))
        with open(fdir / f"{movie}_output.json", "w", encoding="utf-8") as f:
            json.dump([{"interval": r["interval"], "preprocessed_input": r["preprocessed_input"],
                        "summary": r["summary"]} for r in rs], f, indent=2, ensure_ascii=False)

    # ---- report
    def avg(k):
        v = [r[k] for r in rows if r[k] is not None]
        return f"{st.mean(v):.3f}" if v else "-"

    n = len(rows)
    kept = sum(r["keep"] == "yes" for r in rows)
    L = [f"# Summary vs preprocessed text — {inp.name}", "", f"Chunks: {n}", "",
         "| Metric | Average |", "|---|---|"]
    for k, lab in [("faithfulness", "NLI faithfulness (strict, info only)"),
                   ("semantic_sim", "Semantic similarity"),
                   ("rouge1_p", "ROUGE-1 precision vs text"), ("rouge2_p", "ROUGE-2 precision vs text"),
                   ("rougeL_p", "ROUGE-L precision vs text"), ("words", "Words per summary"),
                   ("sentences", "Sentences per summary"), ("compression", "Compression (summary/text words)")]:
        if avg(k) != "-":
            L.append(f"| {lab} | {avg(k)} |")
    if judged:
        yes = sum(r["judge_faithful"] == "yes" for r in rows)
        no = sum(r["judge_faithful"] == "no" for r in rows)
        err = sum(r["judge_faithful"] == "error" for r in rows)
        L += ["", f"## LLM judge ({args.judge_model}, scope: {args.judge_scope}, {judged} chunks judged)", "",
              f"- faithful: **{yes}** ({100 * yes / judged:.1f}%)",
              f"- NOT faithful: **{no}** ({100 * no / judged:.1f}%)  -> dropped",
              f"- judge errors: {err}" + ("  (re-run to retry; successful answers are cached)" if err else "")]
    L += ["", "| Flag | Chunks | % | Action |", "|---|---|---|---|"]
    for fl in ["novel_names", "judge_unfaithful", "too_long", "bad_sentence_count", "low_nli", "off_topic"]:
        c = sum(fl in r["flags"].split(";") for r in rows)
        L.append(f"| {fl} | {c} | {100 * c / n:.1f}% | {'drop' if fl in DROP_FLAGS else 'info'} |")
    L += ["", f"**Kept for training: {kept}/{n} ({100 * kept / n:.1f}%)** -> {fdir}", "",
          "## Worst chunks (dropped first, top 10)", ""]
    for r in rows[:10]:
        L.append(f"- **{r['movie']} {r['interval']}** [{r['flags'] or 'no flag'}] "
                 f"judge={r['judge_faithful'] or '-'} {('— ' + r['judge_unsupported']) if r['judge_unsupported'] else ''}  \n"
                 f"  {r['summary']}")
    report = "\n".join(L)
    (out / "summary_check.md").write_text(report, encoding="utf-8")
    print("\n" + report + f"\n\nFiles: {out.resolve()}")


def data_order(interval):
    """Sort key for 'mm:ss - mm:ss' / 'hh:mm:ss - ...' intervals."""
    try:
        start = interval.split(" - ")[0]
        parts = [float(p) for p in start.replace(",", ".").split(":")]
        sec = 0.0
        for p in parts:
            sec = sec * 60 + p
        return sec
    except Exception:
        return 0.0


if __name__ == "__main__":
    main()
