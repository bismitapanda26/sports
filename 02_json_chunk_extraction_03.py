#!/usr/bin/env python3
"""
02_json_chunk_extraction_03.py

Subtitle cleaning + chunking for on-device subtitle summarization
(teacher: Gemma / Qwen / Gemini, students: t5-efficient-tiny and SmolLM).

Pipeline
--------
  parse (.json/.srt/.vtt) -> drop ad/credit cues -> clean each cue ONCE
  -> drop consecutive duplicate cues -> greedy chunking under
     (a) token budget for EVERY tokenizer given, (b) max time window
  -> split over-long single cues by sentence -> merge tiny chunks
  -> write <movie>_chunk.json

Design rules
------------
* Only ADJACENT stutters inside one cue are collapsed. Repetition across lines
  is real dialogue ("Did you kill him?" / "I didn't kill him.") and is kept.
* Ad/credit filtering is per cue, never per chunk.
* Informative sound cues are kept as tags ([crying], [gunshot], ...);
  pure-noise cues are dropped ((WIND HOWLING), (IN FRENCH)).
* One chunk set is shared by both students (fair comparison): chunk with the
  T5 tokenizer and additionally enforce the SmolLM budget via --check_tokenizer.
* Cleaning is deterministic and minimal: port THIS file's cleaning to the
  device pipeline unchanged to avoid train/inference skew.

Example
-------
python 02_json_chunk_extraction_03.py ./subs ./chunks \
    --tokenizer google/t5-efficient-tiny-dl8 --max_tokens 350 \
    --check_tokenizer HuggingFaceTB/SmolLM-135M --check_max_tokens 700 \
    --max_window_min 10
"""

import argparse
import html
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
DEFAULT_MAX_TOKENS = 350
DEFAULT_MAX_WINDOW_MINUTES = 10.0
DEFAULT_MIN_TOKENS = 16
DEFAULT_TOKENIZER = "google/t5-efficient-tiny-dl8"
DUP_CUE_WINDOW_SEC = 2.0      # identical consecutive cues within this gap -> dropped

# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------
# SRT / VTT timestamp; hours optional (VTT allows mm:ss.mmm)
TIME_RE = re.compile(
    r"(?:(\d{1,2}):)?(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(?:(\d{1,2}):)?(\d{2}):(\d{2})[,.](\d{1,3})"
)

JUNK_RE = re.compile("|".join([
    r"opensubtitles", r"subscene", r"\byify\b", r"\byts\.", r"addic7ed", r"podnapisi",
    r"subtitles? (?:by|ripped|synced|downloaded|provided)", r"sync(?:ed)? (?:by|&)",
    r"corrected by", r"translated by", r"captioning (?:made possible|by)",
    r"www\.", r"https?://", r"\.com\b", r"\.org\b", r"\.net\b", r"advertise your product",
]), re.IGNORECASE)

# Sound cues worth keeping (carry plot / emotion). First match wins.
KEEP_CUES = [
    ("crying",    r"sob|\bcry|cries|crying|weep|bawl|whimper"),
    ("screaming", r"scream|shriek|yell|shout"),
    ("laughter",  r"laugh|chuckl|giggl|cackl"),
    ("cheering",  r"cheer|applau|whoop"),
    ("gunshot",   r"gunshot|gunfire|shots? fired|gun cock"),
    ("explosion", r"explosion|explod|blast"),
    ("sirens",    r"siren"),
    ("music",     r"music|♪|♫|\bsong|singing"),
]
KEEP_CUE_RES = [(name, re.compile(p, re.IGNORECASE)) for name, p in KEEP_CUES]

BRACKET_CUE_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")
STAR_RE = re.compile(r"\*([^*]+)\*")

# Speaker label only at line start, ALL-CAPS: "WOMAN:", "DR. SMITH:", "MAN 2:"
SPEAKER_RE = re.compile(r"^-?\s*([A-Z][A-Z0-9 .'\-]{0,24}[A-Z0-9]):\s*")

# Hesitation fillers. "Uh-huh" / "Mm-hmm" mean "yes" and are kept.
FILLER_RE = re.compile(
    r"(?i)(?<![\w'-])(?:u+h+|u+m+|e+r+m+|m{2,}|h+m{2,})(?![\w']|-(?:huh|hmm|hm))[\s,.…!?\-]*"
)

# Single words that legitimately repeat ("I had had enough", "no, no")
LEGIT_DOUBLE = {"had", "that", "is", "do", "very", "bye", "no", "so", "really", "go", "far", "more"}

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Cue:
    start: float
    end: float
    text: str


# --------------------------------------------------------------------------
# Tokenizers
# --------------------------------------------------------------------------
class TokCounter:
    """Token counter for one tokenizer with its own budget."""

    def __init__(self, name, budget):
        self.name, self.budget, self.tok = name or "word_estimate", budget, None
        if name:
            try:
                from transformers import AutoTokenizer
                self.tok = AutoTokenizer.from_pretrained(name)
                self.tok.model_max_length = 10 ** 9      # silence length warnings
            except Exception as e:
                print(f"⚠️  Could not load tokenizer '{name}': {e}. Using word estimate (x1.3).")
                self.name = f"{name} (word_estimate)"

    def count(self, text):
        if not text:
            return 0
        if self.tok is not None:
            return len(self.tok.encode(text, add_special_tokens=False))
        w = len(text.split())
        return int(w * 1.3) + 1 if w else 0

    def unk(self, text):
        if self.tok is None or self.tok.unk_token_id is None:
            return 0
        return self.tok.encode(text, add_special_tokens=False).count(self.tok.unk_token_id)


# --------------------------------------------------------------------------
# File reading / parsing
# --------------------------------------------------------------------------
def read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        from charset_normalizer import from_bytes   # optional, better guesses
        best = from_bytes(raw).best()
        if best is not None:
            return str(best)
    except ImportError:
        pass
    try:
        return raw.decode("cp1252")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _hms_to_sec(h, m, s, ms) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000.0


def _str_time_to_sec(val: str):
    parts = val.strip().replace(",", ".").split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        if len(parts) == 1:
            return float(parts[0])
    except ValueError:
        return None
    return None


START_KEYS = ["start", "start_sec", "start_time", "startTime", "from", "begin", "time", "timestamp"]
END_KEYS = ["end", "end_sec", "end_time", "endTime", "to"]
TEXT_KEYS = ["text", "content", "subtitle", "dialogue", "sentence", "caption"]


def _first(entry, keys):
    for k in keys:
        if k in entry and entry[k] not in (None, ""):
            return entry[k]
    return None


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def parse_json(text: str):
    data = json.loads(text)
    if isinstance(data, dict):
        for key in ["subtitles", "segments", "dialogue", "lines", "data", "cues"]:
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("JSON root is not a list of cues")

    raw, nums = [], []
    for e in data:
        if not isinstance(e, dict):
            continue
        t = _first(e, TEXT_KEYS)
        if t is None:
            continue
        if isinstance(t, list):
            t = "\n".join(map(str, t))
        s, en = _first(e, START_KEYS), _first(e, END_KEYS)
        nums += [float(v) for v in (s, en) if _is_num(v)]
        raw.append((s, en, str(t)))

    # ms vs seconds decided once per FILE (no film runs 10 h).
    is_ms = bool(nums) and max(nums) > 36000

    def conv(v):
        if v is None:
            return None
        if _is_num(v):
            return float(v) / 1000.0 if is_ms else float(v)
        return _str_time_to_sec(str(v))

    cues = []
    for s, en, t in raw:
        st = conv(s)
        if st is None:
            continue
        et = conv(en)
        if et is None or et < st:
            et = st + 3.0
        cues.append(Cue(st, et, t))
    return cues


def parse_srt_vtt(text: str):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cues = []
    for block in re.split(r"\n\s*\n", text):
        lines = [l for l in block.strip().split("\n") if l.strip()]
        for i, line in enumerate(lines):
            m = TIME_RE.search(line)
            if m:
                g = m.groups()
                body = "\n".join(lines[i + 1:])    # keep line breaks (per-line speakers)
                if body.strip():
                    cues.append(Cue(_hms_to_sec(*g[0:4]), _hms_to_sec(*g[4:8]), body))
                break
    return cues


def parse_subtitles(path: Path):
    text = read_text(path)
    cues = parse_json(text) if path.suffix.lower() == ".json" else parse_srt_vtt(text)
    cues.sort(key=lambda c: c.start)
    return cues


# --------------------------------------------------------------------------
# Cleaning (applied ONCE per cue)
# --------------------------------------------------------------------------
def _cue_tag(name, style):
    return f" ({name}) " if style == "paren" else f" [{name}] "


def _map_cue(inner, style):
    for name, rx in KEEP_CUE_RES:
        if rx.search(inner):
            return _cue_tag(name, style)
    return " "


def _norm_word(tok):
    return re.sub(r"[^\w']", "", tok).lower()


def _trail(tok):
    m = re.search(r"[.?!,]+$|\.\.\.$", tok)
    return m.group(0) if m else ""


def collapse_adjacent_repeats(text):
    """
    Collapse adjacent repeated phrases (1-4 words) within ONE cue, keeping the
    second copy (it carries the final punctuation):
        'Get out! Get out!'            -> 'Get out!'
        "I don't... I don't want to"   -> "I don't want to"
    Not collapsed when the first copy closes a sentence and the second starts a
    new one:  'You want what? What are you doing?'  stays as is.
    """
    toks = text.split()
    changed = True
    while changed:
        changed = False
        for n in (4, 3, 2, 1):
            i, out = 0, []
            while i < len(toks):
                a, b = toks[i:i + n], toks[i + n:i + 2 * n]
                na, nb = [_norm_word(t) for t in a], [_norm_word(t) for t in b]
                if len(a) == n and na == nb and all(na) and not (n == 1 and na[0] in LEGIT_DOUBLE):
                    ta, tb = _trail(a[-1]), _trail(b[-1])
                    sentence_break = ta in (".", "?") and ta != tb
                    if not sentence_break:
                        i += n
                        changed = True
                        continue
                out.append(toks[i])
                i += 1
            toks = out
    return " ".join(toks)


def clean_cue(text, keep_speakers=True, cue_style="bracket"):
    if not text:
        return ""
    t = html.unescape(text)
    t = (t.replace("’", "'").replace("‘", "'").replace("“", '"')
          .replace("”", '"').replace("—", " - ").replace("–", "-")
          .replace(" ", " ").replace(" ", " ").replace("​", ""))
    t = re.sub(r"\{\\[^}]*\}", "", t)     # ASS/SSA override tags {\an8}
    t = re.sub(r"<[^>]+>", "", t)          # <i>, <font>, VTT voice tags

    lines = []
    for line in t.split("\n"):
        line = line.strip()
        if not line:
            continue
        # 1. sound cues -> tag or nothing; *emphasis* unwrapped unless it is a cue
        line = BRACKET_CUE_RE.sub(lambda m: _map_cue(m.group(0)[1:-1], cue_style), line)
        line = STAR_RE.sub(lambda m: _map_cue(m.group(1), cue_style)
                           if any(rx.search(m.group(1)) for _, rx in KEEP_CUE_RES)
                           else m.group(1), line)
        line = re.sub(r"[♪♫]+", _cue_tag("music", cue_style), line)
        line = re.sub(r"(?<!\S)#+(?!\S)", _cue_tag("music", cue_style), line)  # '# lyrics #', keeps '#1'
        # 2. speaker label (line start only), then dialogue dash
        speaker = None
        m = SPEAKER_RE.match(line.strip())
        if m:
            speaker = m.group(1).strip().title()
            line = line.strip()[m.end():]
        line = re.sub(r"^\s*-\s*", "", line)
        # 3. fillers + light normalisation
        line = FILLER_RE.sub(" ", line)
        line = re.sub(r"(?i)\bwanna\b", "want to", line)
        line = re.sub(r"(?i)\bgonna\b", "going to", line)
        line = re.sub(r"(?i)\bgotta\b", "got to", line)
        line = re.sub(r"\.{2,}|…", "...", line)
        line = re.sub(r"\s+([!?,.;:])", r"\1", line)
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"^[,;:.!?\-\s]+", "", line)
        if not line:
            continue                       # e.g. "WOMAN: Uh..." -> nothing left
        if speaker and keep_speakers and re.search(r"[A-Za-z]", re.sub(r"[\[(]\w+[\])]", "", line)):
            line = f"{speaker}: {line}"
        lines.append(line)

    t = " ".join(lines)
    t = collapse_adjacent_repeats(t)
    t = re.sub(r"([\[(]\w+[\])])(?:\s*\1)+", r"\1", t)   # [music] [music] -> [music]
    t = re.sub(r"[\s,;:\-]+$", "", t)
    return t.strip()


def has_substantive_dialogue(text):
    stripped = re.sub(r"[\[(]\w+[\])]", "", text)
    stripped = re.sub(r"\b[A-Z][\w .'-]{0,25}:\s", "", stripped)
    return sum(ch.isalpha() for ch in stripped) >= 3


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def format_timestamp(sec):
    sec = max(0.0, sec)
    h, m, s = int(sec // 3600), int((sec % 3600) // 60), int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _fits(counters, text):
    return all(c.count(text) <= c.budget for c in counters)


def split_long_cue(cue, counters):
    """Split a single cue that alone exceeds a budget, by sentence then by words."""
    if _fits(counters, cue.text):
        return [cue]
    pieces, cur = [], ""
    units = SENT_SPLIT_RE.split(cue.text)
    expanded = []
    for u in units:                         # sentence still too long -> word windows
        if _fits(counters, u):
            expanded.append(u)
        else:
            words, buf = u.split(), []
            for w in words:
                if buf and not _fits(counters, " ".join(buf + [w])):
                    expanded.append(" ".join(buf))
                    buf = []
                buf.append(w)
            if buf:
                expanded.append(" ".join(buf))
    for u in expanded:
        cand = f"{cur} {u}".strip()
        if cur and not _fits(counters, cand):
            pieces.append(cur)
            cur = u
        else:
            cur = cand
    if cur:
        pieces.append(cur)
    dur = (cue.end - cue.start) / max(1, len(pieces))
    return [Cue(cue.start + i * dur, cue.start + (i + 1) * dur, p) for i, p in enumerate(pieces)]


def process_single_file(path, counters, max_window_sec, min_tokens,
                        keep_speakers=True, cue_style="bracket", stats=None):
    stats = stats if stats is not None else {}
    bump = lambda k, n=1: stats.__setitem__(k, stats.get(k, 0) + n)
    primary = counters[0]

    cues = parse_subtitles(path)
    if not cues:
        return []

    # 1. clean each cue once; drop ads / empties / consecutive duplicates
    cleaned, prev_norm, prev_end = [], None, -1e9
    for c in cues:
        if JUNK_RE.search(c.text):
            bump("junk_cues")
            continue
        t = clean_cue(c.text, keep_speakers, cue_style)
        if not t:
            continue
        norm = re.sub(r"\W+", " ", t.lower()).strip()
        if norm == prev_norm and c.start - prev_end <= DUP_CUE_WINDOW_SEC:
            bump("dup_cues")
            prev_end = c.end
            continue
        prev_norm, prev_end = norm, c.end
        for piece in split_long_cue(Cue(c.start, c.end, t), counters):
            if piece.text != t:
                bump("split_cue_pieces")
            cleaned.append(piece)

    # 2. greedy accumulation under all budgets + time window
    chunks, cur = [], []

    def joined(lines):
        return re.sub(r"([\[(]\w+[\])])(?:\s*\1)+", r"\1", " ".join(x.text for x in lines))

    for c in cleaned:
        if cur and _fits(counters, joined(cur + [c])) and c.end - cur[0].start <= max_window_sec:
            cur.append(c)
        else:
            if cur:
                chunks.append(cur)
            cur = [c]
    if cur:
        chunks.append(cur)

    # 3. tiny chunks: merge backward, else carry forward, else drop
    merged, carry = [], []
    for ch in chunks:
        if carry:
            cand = carry + ch
            if _fits(counters, joined(cand)) and cand[-1].end - cand[0].start <= max_window_sec:
                ch = cand
            else:
                bump("dropped_small")
            carry = []
        text = joined(ch)
        if not has_substantive_dialogue(text):
            bump("dropped_no_dialogue")
            continue
        if primary.count(text) >= min_tokens:
            merged.append(ch)
            continue
        if merged:
            cand = merged[-1] + ch
            if _fits(counters, joined(cand)) and cand[-1].end - cand[0].start <= max_window_sec:
                merged[-1] = cand
                bump("merged_small")
                continue
        carry = ch
    if carry:
        bump("dropped_small")

    # 4. emit
    result = []
    for ch in merged:
        text = joined(ch)
        start, end = ch[0].start, max(x.end for x in ch)
        rec = {
            "interval": f"{format_timestamp(start)} - {format_timestamp(end)}",
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "num_tokens": primary.count(text),
        }
        for cnt in counters[1:]:
            rec[f"num_tokens_{cnt.name.split('/')[-1]}"] = cnt.count(text)
        unk = primary.unk(text)
        if unk:
            bump("unk_tokens", unk)
        if not _fits(counters, text):
            bump("over_budget")
        rec["preprocessed_input"] = text
        result.append(rec)
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Clean and chunk subtitle files for summarization.")
    ap.add_argument("input_path", help="Subtitle file or folder (.json/.srt/.vtt)")
    ap.add_argument("output_dir", help="Folder for *_chunk.json outputs")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                    help="Primary tokenizer (student with the tightest input limit). '' = word estimate")
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="Budget for chunk text in primary tokenizer (leave room for prefix + prev summary)")
    ap.add_argument("--check_tokenizer", default="",
                    help="Optional 2nd tokenizer whose budget is also enforced, e.g. HuggingFaceTB/SmolLM-135M")
    ap.add_argument("--check_max_tokens", type=int, default=700,
                    help="Budget for chunk text in 2nd tokenizer (context - prompt - max output)")
    ap.add_argument("--max_window_min", type=float, default=DEFAULT_MAX_WINDOW_MINUTES)
    ap.add_argument("--min_tokens", type=int, default=DEFAULT_MIN_TOKENS)
    ap.add_argument("--no_speakers", action="store_true", help="Drop speaker labels instead of 'Woman: ...'")
    ap.add_argument("--cue_style", choices=["bracket", "paren"], default="bracket",
                    help="[music] or (music). Use 'paren' if the T5 tokenizer reports unk tokens")
    args = ap.parse_args()

    src, out_dir = Path(args.input_path).resolve(), Path(args.output_dir).resolve()
    if not src.exists():
        print(f"❌ Input path '{src}' does not exist.")
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    counters = [TokCounter(args.tokenizer, args.max_tokens)]
    if args.check_tokenizer:
        counters.append(TokCounter(args.check_tokenizer, args.check_max_tokens))

    exts = (".json", ".srt", ".vtt")
    if src.is_file():
        files = [src]
    else:
        files = sorted(p for p in src.rglob("*")
                       if p.suffix.lower() in exts
                       and not p.name.endswith("_chunk.json")
                       and out_dir not in p.resolve().parents)

    print(f"📂 Scanning: {src}\n🔍 Found {len(files)} subtitle files.")
    print("🔢 Budgets: " + ", ".join(f"{c.name} ≤ {c.budget}" for c in counters) +
          f", window ≤ {args.max_window_min} min\n" + "-" * 40)

    totals, n_chunks = {}, 0
    for fp in files:
        stats = {}
        stem = re.sub(r"_cleaned$", "", fp.stem)
        out_path = out_dir / f"{stem}_chunk.json"
        try:
            chunks = process_single_file(fp, counters, args.max_window_min * 60, args.min_tokens,
                                         keep_speakers=not args.no_speakers,
                                         cue_style=args.cue_style, stats=stats)
        except Exception as e:
            print(f"❌ Failed '{fp.name}': {type(e).__name__}: {e}")
            totals["failed_files"] = totals.get("failed_files", 0) + 1
            continue
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
        if chunks:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=4, ensure_ascii=False)
            n_chunks += len(chunks)
            avg = sum(c["num_tokens"] for c in chunks) / len(chunks)
            print(f"✅ {fp.name} → {out_path.name} ({len(chunks)} chunks, avg {avg:.0f} tok) {stats or ''}")
        else:
            print(f"⚠️  Skipped '{fp.name}' (no substantive content) {stats or ''}")

    print("-" * 40 + f"\n🎉 Done. {n_chunks} chunks. Totals: {totals}")
    if totals.get("unk_tokens"):
        print("⚠️  Primary tokenizer produced <unk> tokens; try --cue_style paren and inspect the text.")


if __name__ == "__main__":
    main()
