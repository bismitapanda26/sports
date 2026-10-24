#!/usr/bin/env python3
"""
07_finetune_t5.py

Fine-tune t5-efficient-tiny for on-device subtitle summarization
(distillation from teacher summaries).

Data: train.jsonl / val.jsonl from make_split.py, one record per line:
      {"movie":..., "interval":..., "input": <preprocessed text>, "target": <summary>}

PROMPT HANDLING
---------------
The task prompt is stored as METADATA in the saved model's config.json under the key
"subtitle_summarization" (also copied to <model>/prompt_meta.json):
    {"prompt": "...", "use_prefix": false, "prefix": "", "max_source_len": ..., "generation": {...}}

* Default (--use_prefix off): the model is trained on the plain preprocessed
  text. At inference you pass ONLY the text - nothing else. The prompt is kept
  in the config purely for documentation / reproducibility.
* --use_prefix: "<prefix>" is prepended during training, and the SAME prefix
  is recorded in the config; predict mode (and your device code) must read it
  from there and prepend it. Train and inference inputs must always match.

CHECKPOINT SELECTION
--------------------
--select_metric cosine (default): best epoch = highest mean cosine similarity between
student and teacher summaries (sentence embeddings; meaning, not exact words).
Falls back to rougeL if sentence-transformers is not installed.
During training, validation uses greedy decoding (fast); the final validation
predictions use the full beam search settings stored with the model.

TRAIN
-----
python 07_finetune_t5.py train --train data_baseline/train.jsonl --val data_baseline/val.jsonl \
    --out runs/t5_baseline

PREDICT (reads prefix / generation settings from the saved config; you pass text only)
-------
python 07_finetune_t5.py predict --model runs/t5_baseline/best --input test.jsonl \
    --output preds.jsonl
"""

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, T5ForConditionalGeneration,
                          set_seed)

META_KEY = "subtitle_summarization"
DEFAULT_PROMPT = ("Summarize the movie subtitle dialogue in 2-3 sentences (max 50 words), third person, "
                  "present tense. Use names only if they appear in the dialogue; otherwise use relationships "
                  "(e.g. 'a mother and her son') or neutral descriptions. Do not invent events.")
DEFAULT_GEN = {"max_new_tokens": 80, "num_beams": 4, "no_repeat_ngram_size": 3,
               "length_penalty": 1.0, "early_stopping": True}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def read_jsonl(path, need_target=False):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                src = r.get("input") or r.get("preprocessed_input") or ""
                tgt = r.get("target") or r.get("summary") or ""
                if src.strip() and (tgt.strip() or not need_target):
                    rows.append({**r, "input": src, "target": tgt})
    return rows


class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tok, prefix, max_src, max_tgt):
        self.items, self.truncated, self.unk = [], 0, 0
        eos = tok.eos_token_id
        for r in rows:
            ids = tok(prefix + r["input"])["input_ids"]           # tokenize once
            if len(ids) > max_src:                                 # truncate, keep </s>
                self.truncated += 1
                ids = ids[:max_src - 1] + [eos]
            if tok.unk_token_id is not None:
                self.unk += ids.count(tok.unk_token_id)
            tgt = tok(text_target=r["target"])["input_ids"]
            if len(tgt) > max_tgt:
                tgt = tgt[:max_tgt - 1] + [eos]
            self.items.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": tgt})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


# --------------------------------------------------------------------------
# Metrics (no extra dependency): ROUGE-1/2/L F1 + length
# --------------------------------------------------------------------------
def _toks(t):
    return re.findall(r"[a-z0-9']+", t.lower())


def _ngram_f(a, b, n):
    ga = [tuple(a[i:i + n]) for i in range(len(a) - n + 1)]
    gb = [tuple(b[i:i + n]) for i in range(len(b) - n + 1)]
    if not ga or not gb:
        return 0.0
    cb = {}
    for g in gb:
        cb[g] = cb.get(g, 0) + 1
    ov = 0
    for g in ga:
        if cb.get(g, 0):
            cb[g] -= 1
            ov += 1
    if not ov:
        return 0.0
    p, r = ov / len(ga), ov / len(gb)
    return 2 * p * r / (p + r)


def _lcs_f(a, b):
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    l = prev[-1]
    if not l:
        return 0.0
    p, r = l / len(a), l / len(b)
    return 2 * p * r / (p + r)


def rouge_scores(preds, refs):
    r1 = r2 = rl = 0.0
    for p, r in zip(preds, refs):
        tp, tr = _toks(p), _toks(r)
        r1 += _ngram_f(tp, tr, 1)
        r2 += _ngram_f(tp, tr, 2)
        rl += _lcs_f(tp, tr)
    n = max(1, len(preds))
    lp = sum(len(p.split()) for p in preds) / n
    lr = sum(len(r.split()) for r in refs) / n
    return {"rouge1": r1 / n, "rouge2": r2 / n, "rougeL": rl / n,
            "gen_words": lp, "len_ratio": lp / max(1e-9, lr)}


class CosineScorer:
    """Mean cosine similarity of sentence embeddings (student vs teacher)."""
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)

    def __call__(self, preds, refs):
        a = self.m.encode(preds, convert_to_tensor=True, normalize_embeddings=True, batch_size=64)
        b = self.m.encode(refs, convert_to_tensor=True, normalize_embeddings=True, batch_size=64)
        return float((a * b).sum(-1).mean())


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
def make_training_args(**kw):
    """Seq2SeqTrainingArguments across transformers versions (drops/renames unknown args)."""
    import inspect
    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "evaluation_strategy" in params:   # transformers < 4.41
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    dropped = [k for k in kw if k not in params]
    if dropped:
        print(f"[info] TrainingArguments not supported by this transformers version, ignored: {dropped}")
    return Seq2SeqTrainingArguments(**{k: v for k, v in kw.items() if k in params})


def make_trainer(**kw):
    """Seq2SeqTrainer across transformers 4.x / 5.x (tokenizer -> processing_class)."""
    tok = kw.pop("tok")
    try:
        return Seq2SeqTrainer(processing_class=tok, **kw)
    except TypeError:
        return Seq2SeqTrainer(tokenizer=tok, **kw)


def train(args):
    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    prefix = args.prefix if args.use_prefix else ""

    tr_rows, va_rows = read_jsonl(args.train, need_target=True), read_jsonl(args.val, need_target=True)
    tr_movies = {r.get("movie") for r in tr_rows if r.get("movie")}
    va_movies = {r.get("movie") for r in va_rows if r.get("movie")}
    overlap = tr_movies & va_movies
    if overlap:
        print(f"[WARN] {len(overlap)} movies are in BOTH train and val (leakage): {sorted(overlap)[:5]} ...")
    train_ds = Seq2SeqDataset(tr_rows, tok, prefix, args.max_source_len, args.max_target_len)
    val_ds = Seq2SeqDataset(va_rows, tok, prefix, args.max_source_len, args.max_target_len)
    print(f"train: {len(train_ds)} chunks from {len(tr_movies)} movies ({train_ds.truncated} inputs truncated "
          f"at {args.max_source_len} tokens) | val: {len(val_ds)} chunks from {len(va_movies)} movies "
          f"({val_ds.truncated} truncated)")
    if train_ds.unk:
        print(f"[WARN] {train_ds.unk} <unk> tokens in training inputs - check characters the T5 vocab lacks "
              f"(e.g. use --cue_style paren in preprocessing if [ ] become <unk>)")
    tgt_lens = [len(x["labels"]) for x in train_ds.items]
    p95 = np.percentile(tgt_lens, 95)
    print(f"target tokens: mean {np.mean(tgt_lens):.0f}, p95 {p95:.0f}, max {max(tgt_lens)} "
          f"(cap {args.max_target_len}, generation max_new_tokens {args.gen_max_new_tokens})")
    if p95 > args.gen_max_new_tokens:
        print(f"[WARN] 5% of targets are longer than --gen_max_new_tokens; summaries may be cut off")

    cos_scorer = None
    if args.select_metric == "cosine":
        try:
            cos_scorer = CosineScorer(args.embed_model)
        except Exception as e:
            print(f"[WARN] cosine scorer unavailable ({e}); selecting best checkpoint by rougeL")
            args.select_metric = "rougeL"

    gen = dict(DEFAULT_GEN)
    gen["max_new_tokens"] = args.gen_max_new_tokens
    gen["num_beams"] = args.num_beams
    model.generation_config.update(**gen)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds < 0, tok.pad_token_id, preds)
        labels = np.where(labels < 0, tok.pad_token_id, labels)
        p = [x.strip() for x in tok.batch_decode(preds, skip_special_tokens=True)]
        r = [x.strip() for x in tok.batch_decode(labels, skip_special_tokens=True)]
        m = rouge_scores(p, r)
        if cos_scorer:
            m["cosine"] = cos_scorer([x or "." for x in p], r)
        return {k: round(v, 4) for k, v in m.items()}

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    warmup_steps = max(1, int(0.05 * steps_per_epoch * args.epochs))
    targs = make_training_args(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        num_train_epochs=args.epochs,
        label_smoothing_factor=args.label_smoothing,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=args.select_metric,
        greater_is_better=True,
        predict_with_generate=True,
        generation_num_beams=args.eval_beams,      # fast greedy eval during training
        logging_steps=50,
        bf16=bf16,
        fp16=False,                      # T5 is unstable in fp16
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
        optim=args.optim,
    )
    trainer = make_trainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, tok=tok,
        data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()

    # ---- save best model + prompt metadata
    best_dir = Path(args.out) / "best"
    meta = {
        "prompt": args.prompt,                 # stored for reference only
        "use_prefix": bool(args.use_prefix),
        "prefix": prefix,                      # "" -> pass plain preprocessed text at inference
        "max_source_len": args.max_source_len,
        "max_target_len": args.max_target_len,
        "generation": gen,
        "train_file": str(args.train),
        "val_file": str(args.val),
        "base_model": args.model,
        "select_metric": args.select_metric,
    }
    setattr(trainer.model.config, META_KEY, meta)          # serialized into config.json
    trainer.save_model(str(best_dir))
    tok.save_pretrained(str(best_dir))
    with open(best_dir / "prompt_meta.json", "w", encoding="utf-8") as f:   # same info, easy to read on device side
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nBest model saved to {best_dir}\nMetadata ({META_KEY}) in config.json: "
          f"use_prefix={meta['use_prefix']} prefix={meta['prefix']!r}")

    # ---- final validation report + sample predictions
    metrics = trainer.evaluate()
    print({k: v for k, v in metrics.items() if k.startswith("eval_")})
    preds = predict_rows(trainer.model, tok, va_rows, meta, batch_size=args.batch_size)
    with open(Path(args.out) / "val_predictions.jsonl", "w", encoding="utf-8") as f:
        for r, p in zip(va_rows, preds):
            f.write(json.dumps({"movie": r.get("movie"), "interval": r.get("interval"), "input": r["input"],
                                "teacher": r["target"], "student": p}, ensure_ascii=False) + "\n")
    final = rouge_scores(preds, [r["target"] for r in va_rows])
    if cos_scorer:
        final["cosine"] = cos_scorer([x or "." for x in preds], [r["target"] for r in va_rows])
    with open(Path(args.out) / "val_metrics.json", "w") as f:
        json.dump({"during_training_greedy": metrics,
                   "final_beam_search": {k: round(v, 4) for k, v in final.items()}}, f, indent=2)
    with open(Path(args.out) / "run_config.json", "w") as f:
        json.dump({k: v for k, v in vars(args).items()}, f, indent=2)
    print(f"Final validation (beam search): { {k: round(v, 4) for k, v in final.items()} }")
    print(f"Validation predictions -> {Path(args.out) / 'val_predictions.jsonl'}")


# --------------------------------------------------------------------------
# Predict (prefix + generation settings come from the saved config)
# --------------------------------------------------------------------------
def load_meta(model, model_dir=None):
    meta = getattr(model.config, META_KEY, None)
    if not meta and model_dir and (Path(model_dir) / "prompt_meta.json").exists():
        meta = json.load(open(Path(model_dir) / "prompt_meta.json", encoding="utf-8"))
    return meta or {"prefix": "", "max_source_len": 512, "generation": DEFAULT_GEN}


@torch.no_grad()
def predict_rows(model, tok, rows, meta, batch_size=16):
    device = next(model.parameters()).device
    model.eval()
    prefix = meta.get("prefix", "")
    gen = dict(meta.get("generation", DEFAULT_GEN))
    out = []
    for i in range(0, len(rows), batch_size):
        texts = [prefix + r["input"] for r in rows[i:i + batch_size]]   # caller supplies text only
        enc = tok(texts, max_length=meta.get("max_source_len", 512), truncation=True,
                  padding=True, return_tensors="pt").to(device)
        ids = model.generate(**enc, **gen)
        out += [t.strip() for t in tok.batch_decode(ids, skip_special_tokens=True)]
    return out


def predict(args):
    tok = AutoTokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    if torch.cuda.is_available():
        model.cuda()
    meta = load_meta(model, args.model)
    print(f"prefix from config: {meta.get('prefix', '')!r} (use_prefix={meta.get('use_prefix', False)})")
    rows = read_jsonl(args.input)
    preds = predict_rows(model, tok, rows, meta, args.batch_size)
    with open(args.output, "w", encoding="utf-8") as f:
        for r, p in zip(rows, preds):
            f.write(json.dumps({"movie": r.get("movie"), "interval": r.get("interval"),
                                "input": r["input"], "summary": p,
                                **({"teacher": r["target"]} if r.get("target") else {})},
                               ensure_ascii=False) + "\n")
    if all(r.get("target") for r in rows):
        print({k: round(v, 4) for k, v in rouge_scores(preds, [r["target"] for r in rows]).items()})
    print(f"{len(preds)} predictions -> {args.output}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fine-tune / run t5-efficient-tiny for subtitle summarization.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--train", required=True)
    t.add_argument("--val", required=True)
    t.add_argument("--out", default="runs/t5_tiny")
    t.add_argument("--model", default="google/t5-efficient-tiny-dl8")
    t.add_argument("--prompt", default=DEFAULT_PROMPT, help="Stored in config metadata")
    t.add_argument("--use_prefix", action="store_true",
                   help="Also prepend --prefix to inputs (then inference must use it too)")
    t.add_argument("--prefix", default="summarize: ")
    t.add_argument("--max_source_len", type=int, default=512)
    t.add_argument("--max_target_len", type=int, default=128)
    t.add_argument("--gen_max_new_tokens", type=int, default=80)
    t.add_argument("--num_beams", type=int, default=4, help="Beams for final predictions / deployment")
    t.add_argument("--eval_beams", type=int, default=1, help="Beams for per-epoch validation (1 = fast greedy)")
    t.add_argument("--select_metric", choices=["cosine", "rougeL", "rouge1", "rouge2"], default="cosine",
                   help="Validation metric used to pick the best checkpoint / early stopping")
    t.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Sentence-embedding model for the cosine metric")
    t.add_argument("--optim", default="adamw_torch", help="adamw_torch or adafactor (classic for T5)")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch_size", type=int, default=32)
    t.add_argument("--grad_accum", type=int, default=1)
    t.add_argument("--lr", type=float, default=1e-3, help="Tiny T5 tolerates a high LR; try 5e-4 if unstable")
    t.add_argument("--label_smoothing", type=float, default=0.1)
    t.add_argument("--patience", type=int, default=4, help="Early stopping patience (epochs without improvement)")
    t.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("predict")
    p.add_argument("--model", required=True, help="Saved model dir, e.g. runs/t5_tiny/best")
    p.add_argument("--input", required=True, help="jsonl with 'input' (or 'preprocessed_input')")
    p.add_argument("--output", required=True)
    p.add_argument("--batch_size", type=int, default=16)

    args = ap.parse_args()
    train(args) if args.cmd == "train" else predict(args)


if __name__ == "__main__":
    main()
