#!/usr/bin/env python3
"""
03_gemma_summary_generator.py

Generate teacher (reference) summaries for chunk files produced by
02_json_chunk_extraction_03.py, using any OpenAI-compatible chat API
(Gemma / Qwen / Gemini via LiteLLM, vLLM, Ollama ...).

Key behaviour
-------------
* Deterministic labels (temperature 0 + seed) so runs are reproducible.
* Thinking/reasoning text is NEVER used as a summary.
* Failed chunks are written with status="error" and an EMPTY summary
  (filter on status == "ok" before training).
* Output length is capped and validated (sentences / words); bad outputs
  are retried, and still-bad ones are marked status="invalid".
* Results are appended to <movie>_output.jsonl after every chunk, so a
  crash loses nothing; re-running resumes from where it stopped.
* Every chunk is summarized INDEPENDENTLY: no previous summary and no chat
  history is ever sent, so nothing leaks from earlier chunks.
* Names are checked against the dialogue: a proper noun (person / place /
  title) that does not appear in the source text triggers a rewrite; if it
  is still there after the retries, the chunk is marked invalid and excluded.

Example
-------
python 03_gemma_summary_generator.py -i ./chunks -o ./summaries \
    --model google/gemma-4-31B --api_url http://<server>:4000/v1/chat/completions
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# API configuration (key handling unchanged: env var > fallback)
# --------------------------------------------------------------------------
DEFAULT_API_URL = "http://107.108.92.242:4000/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-4-31B"
DEFAULT_FALLBACK_KEY = ""                 # <-- put your fallback key here
API_KEY = os.getenv("GEMMA_API_KEY") or DEFAULT_FALLBACK_KEY

# --------------------------------------------------------------------------
# Generation parameters
# --------------------------------------------------------------------------
TEMPERATURE = 0.0
SEED = 42
MAX_TOKENS = 160          # 2-3 sentences; keeps labels inside the student's 200-token output cap
REQUEST_TIMEOUT = 180     # seconds
MAX_RETRIES = 4           # network / server errors
MAX_FORMAT_RETRIES = 2    # outputs that fail validation

MIN_SENTENCES, MAX_SENTENCES = 1, 3
MAX_WORDS = 60

PROMPT_VERSION = "v3"

SYSTEM_PROMPT = """You are an expert movie summarizer. You summarize short segments of movie subtitle dialogue.

Input format:
- Text is subtitle dialogue from one segment of a movie.
- "Name:" at the start of a sentence is a speaker label (e.g. "Woman:", "John:").
- Tags in square brackets are sound cues, not dialogue: [music], [crying], [screaming], [laughter], [cheering], [gunshot], [explosion], [sirens].

Rules:
1. CONCISENESS: Output 2-3 sentences, at most 50 words in total, covering only the key plot points, events, or character actions.
2. NAMES: Never write a proper noun (person, place, organization, title) unless that exact word appears in the dialogue or speaker labels. If unsure, use a neutral description ("a woman", "the man", "his friend", "the city").
3. GROUNDING: Use only this dialogue. Do not try to identify the movie, do not use outside knowledge of it, do not refer to earlier or later scenes, and do not invent events.
4. STYLE: Third person, present tense, plain prose. No quotes from the dialogue, no bullet points, no markdown, no preface such as "Here is the summary".
5. If the dialogue has little plot content, briefly describe what the characters are talking about."""

USER_TEMPLATE = "Dialogue:\n{text}\n\nSummary:"

# Capitalised words that are not names (sentence starters, pronouns, titles, days ...)
NOT_NAMES = set("""
I A An The He She They It We You His Her Their Its Our Your Him Them This That These Those There Here Then
When While After Before As At In On Of For With Without From To Into Meanwhile However Later Finally Also
Both Two Three One Another Some Several Someone Somebody Everyone Nobody Each Others
Mr Mrs Ms Miss Dr Sir Madam Mom Mum Dad Mother Father Grandma Grandpa God Lord Okay OK Yes No
Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April June July
Tension Things Everything Nothing Something Life Love Music Conversation Discussion Talk Danger Fear Panic
Chaos Trouble News Time Night Morning Evening Police Officer Doctor Captain Boss Agent Detective
August September October November December Christmas English French German Spanish
""".split())
_STARTER_VERBS = set("""asks tells says argues discusses explains reveals warns admits insists refuses
agrees accuses confronts meets talks calls tries wants needs plans decides learns finds realizes
describes questions urges begs offers threatens apologizes complains jokes""".split())

PREFACE_RE = re.compile(
    r"^\s*(?:here(?:'s| is) (?:a |the )?(?:concise )?summary[^:\n]*:|summary\s*:|in this (?:segment|scene)[,:]?\s*(?=\w))\s*",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# API call
# --------------------------------------------------------------------------
class RetryableError(Exception):
    pass


_UNSUPPORTED = set()      # params the server rejected once; not sent again


def _headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        k = API_KEY.strip()
        if k.lower().startswith("bearer "):
            k = k[7:].strip()
        h["Authorization"] = f"Bearer {k}"
    return h


def call_api(api_url, model, messages, extra_body=None):
    """POST with timeout + exponential backoff. Returns (content, finish_reason)."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
        "stream": False,
    }
    if extra_body:
        payload.update(extra_body)
    for k in _UNSUPPORTED:
        payload.pop(k, None)

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(api_url, headers=_headers(), json=payload, timeout=REQUEST_TIMEOUT)
            if r.status_code in (408, 429) or r.status_code >= 500:
                raise RetryableError(f"HTTP {r.status_code}: {r.text[:200]}")
            if r.status_code >= 400:
                # Some servers reject unknown params (seed / chat_template_kwargs): drop and retry once.
                for opt in ("chat_template_kwargs", "seed"):
                    if opt in payload:
                        payload.pop(opt)
                        _UNSUPPORTED.add(opt)
                        print(f"    server rejected '{opt}', retrying without it")
                        break
                else:
                    r.raise_for_status()
                continue
            result = r.json()
            choices = result.get("choices") or []
            if not choices:
                raise RetryableError(f"No choices in response: {str(result)[:200]}")
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if content is None:
                content = choices[0].get("text")   # plain completion schema only, never reasoning
            return (content or "").strip(), choices[0].get("finish_reason")
        except (requests.ConnectionError, requests.Timeout, RetryableError) as e:
            last_err = e
            wait = min(60, 2 ** attempt * 3) + random.uniform(0, 1)
            print(f"    retry {attempt + 1}/{MAX_RETRIES} in {wait:.0f}s ({type(e).__name__}: {str(e)[:120]})")
            time.sleep(wait)
    raise RuntimeError(f"API failed after {MAX_RETRIES} attempts: {last_err}")


# --------------------------------------------------------------------------
# Post-processing / validation
# --------------------------------------------------------------------------
def clean_summary(text):
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)  # stray thinking blocks
    t = re.sub(r"[*_#`>]+", "", t)                 # markdown
    t = re.sub(r"^\s*[-•]\s*", "", t, flags=re.MULTILINE)
    t = PREFACE_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip().strip('"').strip("'").strip()
    if t and t[-1] not in ".!?":
        t += "."
    return t


_NOUN_SUFFIX = ("ion", "ness", "ment", "ity", "ence", "ance", "ship", "ing", "ism", "hood", "ure", "ics")
_NLP = None


def _spacy():
    """Optional spaCy NER (pip install spacy && python -m spacy download en_core_web_sm)."""
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        except Exception:
            _NLP = False
    return _NLP or None


def _src_words(source):
    return {re.sub(r"'s?$", "", w) for w in re.findall(r"[a-z][a-z'\-]*", source.lower())}


def novel_names(summary, source):
    """
    Proper nouns in the summary that do not appear as a WHOLE WORD in the dialogue.
    Uses spaCy NER when installed, plus a capitalisation heuristic.
    """
    src = _src_words(source)
    found = []

    def add(word):
        base = re.sub(r"'s?$", "", word)
        if (len(base) >= 2 and base[:1].isupper() and base not in NOT_NAMES
                and base.lower() not in src and base not in found):
            found.append(base)

    nlp = _spacy()
    if nlp:
        for ent in nlp(summary).ents:
            if ent.label_ in ("PERSON", "GPE", "LOC", "ORG", "NORP", "FAC", "WORK_OF_ART", "EVENT", "PRODUCT"):
                for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", ent.text):
                    add(w)

    for sent in re.split(r"(?<=[.!?])\s+", summary):
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", sent)
        for i, w in enumerate(words):
            if not w[:1].isupper():
                continue
            if i == 0:
                # Sentence-initial capital is ambiguous. Treat as a name if:
                #   possessive ("Maria's"), followed by and/or/capitalised word ("Maria and Tom"),
                #   or followed by a verb-like word ending in -s ("Maria reveals") and the word
                #   itself does not look like a common noun ("Tension rises").
                nxt = words[1] if len(words) > 1 else ""
                low = w.lower()
                is_name = (w.endswith("'s")
                           or nxt.lower() in ("and", "or", "with")
                           or (nxt[:1].isupper())
                           or nxt.lower() in _STARTER_VERBS
                           or (nxt.endswith("s") and nxt.islower() and not low.endswith(_NOUN_SUFFIX)))
                if not is_name:
                    continue
            add(w)
    return found


def validate(summary, finish_reason, source=""):
    if not summary:
        return "empty"
    if finish_reason == "length":
        return "truncated"
    n_sent = len([s for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()])
    n_words = len(summary.split())
    if n_sent < MIN_SENTENCES or n_sent > MAX_SENTENCES:
        return f"sentences={n_sent}"
    if n_words > MAX_WORDS:
        return f"words={n_words}"
    names = novel_names(summary, source) if source else []
    if names:
        return "names_not_in_dialogue=" + ",".join(names)
    return None


def get_summary(text, api_url, model, extra_body):
    """Each chunk is a fresh, independent request (system + this dialogue only)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ]
    problem, summary = None, ""
    for attempt in range(MAX_FORMAT_RETRIES + 1):
        raw, finish = call_api(api_url, model, messages, extra_body)
        summary = clean_summary(raw)
        problem = validate(summary, finish, text)
        if problem is None:
            return summary, "ok", None, attempt
        if attempt < MAX_FORMAT_RETRIES:
            # remind the model of the format; temperature stays 0, so the extra turn changes the output
            if problem.startswith("names_not_in_dialogue="):
                bad = problem.split("=", 1)[1].replace(",", ", ")
                fix = (f"The names {bad} do not appear in the dialogue. Rewrite the summary without them, "
                       f"using neutral descriptions (e.g. 'a woman', 'the man'). Only use names that "
                       f"appear in the dialogue. 2-3 sentences, at most 50 words.")
            else:
                fix = (f"That violates the rules ({problem}). Rewrite it as 2-3 sentences, "
                       f"at most 50 words, plain prose only.")
            messages = messages[:2] + [
                {"role": "assistant", "content": raw or "(empty)"},
                {"role": "user", "content": fix},
            ]
    return summary, "invalid", problem, MAX_FORMAT_RETRIES


# --------------------------------------------------------------------------
# Chunk-file processing (resumable JSONL)
# --------------------------------------------------------------------------
def _parse_time_str(t_str):
    try:
        parts = [float(p) for p in t_str.strip().replace(",", ".").split(":")]
    except ValueError:
        return None
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 1:
        return parts[0]
    return None


def _load_done(jsonl_path):
    done = {}
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done[rec["chunk_index"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue        # partial last line after a crash
    return done


def process_chunk_file(input_path, out_dir, args, extra_body):
    print(f"\nProcessing: {input_path.name}")
    with open(input_path, encoding="utf-8") as f:
        chunks = json.load(f)
    if not chunks or not any(k in chunks[0] for k in ("interval", "window_label", "preprocessed_input")):
        print("  [ERROR] Not a chunk file. Run 02_json_chunk_extraction_03.py first.")
        return

    stem = re.sub(r"_chunk$", "", input_path.stem)
    jsonl_path = out_dir / f"{stem}_output.jsonl"
    json_path = out_dir / f"{stem}_output.json"
    if args.overwrite:
        for p in (jsonl_path, json_path):
            if p.exists():
                p.unlink()
    done = _load_done(jsonl_path)
    if args.retry_failed:
        done = {k: v for k, v in done.items() if v.get("status") == "ok"}

    counts = {"ok": 0, "invalid": 0, "error": 0, "skipped": 0, "cached": 0}
    new_records = []

    for i, c in enumerate(chunks):
        source_text = c.get("preprocessed_input") or c.get("text") or c.get("source_text") or ""
        label = c.get("interval") or c.get("window_label") or f"chunk_{i}"
        start = c.get("start_sec", c.get("window_start_sec"))
        end = c.get("end_sec", c.get("window_end_sec"))
        if (start is None or end is None) and " - " in label:
            a, b = label.split(" - ", 1)
            start, end = _parse_time_str(a), _parse_time_str(b)

        if i in done:
            counts["cached"] += 1
            continue
        if not source_text.strip():
            counts["skipped"] += 1
            continue

        rec = {
            "chunk_index": i,
            "window_label": label,
            "window_start_sec": start,
            "window_end_sec": end,
            "source_text": source_text,
            "reference_model": args.model,
            "prompt_version": PROMPT_VERSION,
            "gen_params": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "seed": SEED},
        }
        try:
            summary, status, problem, n_retry = get_summary(source_text, args.api_url, args.model, extra_body)
            rec.update(reference_summary=summary if status == "ok" else "",
                       rejected_output=summary if status != "ok" else None,
                       status=status, problem=problem, format_retries=n_retry)
            print(f"  [{label}] {status}: {summary[:90]}{'...' if len(summary) > 90 else ''}")
        except Exception as e:
            rec.update(reference_summary="", status="error", problem=f"{type(e).__name__}: {e}")
            print(f"  [{label}] ERROR: {e}")
        counts[rec["status"]] += 1

        with open(jsonl_path, "a", encoding="utf-8") as f:   # append immediately: crash-safe
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        new_records.append(rec)

    # Final JSON: only successful chunks, only interval / preprocessed_input / summary.
    # (Full metadata incl. failures stays in the .jsonl log, used for resume / --retry_failed.)
    merged = _load_done(jsonl_path)
    final = [
        {
            "interval": merged[k]["window_label"],
            "preprocessed_input": merged[k]["source_text"],
            "summary": merged[k]["reference_summary"],
        }
        for k in sorted(merged) if merged[k].get("status") == "ok"
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    n_bad = len(merged) - len(final)
    print(f"  Saved {len(final)} summaries -> {json_path.name}"
          f"{f'  ({n_bad} failed, see .jsonl; rerun with --retry_failed)' if n_bad else ''}  this run: {counts}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate teacher summaries for chunked subtitle JSON files.")
    ap.add_argument("-i", "--input", dest="input_target", required=True,
                    help="Chunk JSON file or folder of *_chunk.json files")
    ap.add_argument("-o", "--output", dest="output_dir", required=True, help="Output folder")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Teacher model name on the server")
    ap.add_argument("--api_url", default=DEFAULT_API_URL, help="OpenAI-compatible /chat/completions URL")
    ap.add_argument("--no_thinking_flag", action="store_true",
                    help="Do not send chat_template_kwargs.enable_thinking=false")
    ap.add_argument("-f", "--overwrite", action="store_true", help="Discard existing outputs and regenerate")
    ap.add_argument("--retry_failed", action="store_true", help="Regenerate only chunks not marked ok")
    args = ap.parse_args()

    target, out_dir = Path(args.input_target), Path(args.output_dir)
    if not target.exists():
        print(f"[ERROR] Input '{target}' does not exist.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    if target.is_file():
        files = [target]
    else:
        files = sorted(target.glob("*_chunk.json"))
    if not files:
        print(f"[WARN] No *_chunk.json files found in '{target}'.")
        return

    extra_body = None if args.no_thinking_flag else {"chat_template_kwargs": {"enable_thinking": False}}
    print(f"Model: {args.model} | independent chunks | prompt {PROMPT_VERSION} | {len(files)} files")
    for fp in files:
        process_chunk_file(fp, out_dir, args, extra_body)


if __name__ == "__main__":
    main()
