#!/usr/bin/env python3
"""
09_finetune_smollm.py

Fine-tune SmolLM (decoder-only, e.g. HuggingFaceTB/SmolLM-135M) for subtitle
summarization on the SAME train.jsonl / val.jsonl as the T5 run.

Sequence layout (one causal sequence):
    <template prefix> <dialogue> <template suffix> <summary> <eos>
    e.g. "Dialogue:\\n{input}\\nSummary:"  +  " A mother tells her son ..." + </s>
Loss is computed ONLY on the summary tokens (+ eos); prompt / dialogue tokens are masked.

PROMPT / TEMPLATE
-----------------
A decoder-only model needs a fixed template so it knows where the summary starts.
Default template is minimal (no instruction text, ~5 tokens): the behaviour is learned
from the labels. It is saved in <out>/best/prompt_meta.json and config.json
("subtitle_summarization"). The device code (llama.cpp) MUST build exactly the same
string: prefix + dialogue + suffix, then generate until eos.

NO-TEMPLATE MODE (nothing passed at inference except the dialogue):
    --template "{input}" --target_prefix newline
    sequence = <dialogue> + "\n" + <summary> + <eos>, loss only on "\n" + summary + eos.
    The model learns "after the dialogue, write a newline and the summary"; at inference
    you feed only the dialogue tokens and it generates the newline + summary itself.
    Compare its validation scores with the default template before choosing.

Long dialogues: only the DIALOGUE is truncated (from the end) so that
template + dialogue + summary fit in --max_len. The same dialogue budget
(max_input_tokens) is saved in prompt_meta.json for the device side.

Checkpoint selection: every epoch the model generates summaries for a val subset;
cosine similarity to the teacher (sentence embeddings) picks the best epoch and drives
early stopping (falls back to eval loss if sentence-transformers is missing).

TRAIN
-----
python 09_finetune_smollm.py train --train data_baseline/train.jsonl --val data_baseline/val.jsonl \\
    --out runs/smollm_baseline

PREDICT
-------
python 09_finetune_smollm.py predict --model runs/smollm_baseline/best --input test.jsonl --output preds.jsonl
"""

import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, Trainer,
                          TrainingArguments, set_seed)

META_KEY = "subtitle_summarization"
DEFAULT_TEMPLATE = "Dialogue:\n{input}\nSummary:"
DEFAULT_GEN = {"max_new_tokens": 80, "do_sample": False, "num_beams": 1,
               "repetition_penalty": 1.1, "no_repeat_ngram_size": 3}


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
                    rows.append({**r, "input": src, "target": tgt.strip()})
    return rows


class Template:
    """Tokenizes prefix / dialogue / suffix separately so train and inference match exactly."""

    def __init__(self, tok, template, max_input_tokens):
        if "{input}" not in template:
            raise ValueError("template must contain {input}")
        self.tok, self.template = tok, template
        self.prefix, self.suffix = template.split("{input}", 1)
        self.pre_ids = tok(self.prefix, add_special_tokens=False)["input_ids"] if self.prefix else []
        self.suf_ids = tok(self.suffix, add_special_tokens=False)["input_ids"] if self.suffix else []
        self.max_input_tokens = max_input_tokens
        self.bos = [tok.bos_token_id] if (tok.bos_token_id is not None and
                                          getattr(tok, "add_bos_token", False)) else []

    def prompt_ids(self, text):
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        truncated = len(ids) > self.max_input_tokens
        ids = ids[:self.max_input_tokens]
        return self.bos + self.pre_ids + ids + self.suf_ids, truncated


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tok, tpl, target_prefix=" "):
        self.items, self.truncated = [], 0
        eos = tok.eos_token_id
        for r in rows:
            p_ids, trunc = tpl.prompt_ids(r["input"])
            self.truncated += trunc
            t_ids = tok(target_prefix + r["target"], add_special_tokens=False)["input_ids"] + [eos]
            self.items.append({"input_ids": p_ids + t_ids,
                               "labels": [-100] * len(p_ids) + t_ids})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collator(pad_id):
    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            lab[i, :L] = torch.tensor(b["labels"])
            att[i, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return collate


# --------------------------------------------------------------------------
# Metrics
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
    return 0.0 if not ov else 2 * (ov / len(ga)) * (ov / len(gb)) / (ov / len(ga) + ov / len(gb))


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
    return 0.0 if not l else 2 * (l / len(a)) * (l / len(b)) / (l / len(a) + l / len(b))


def rouge_scores(preds, refs):
    n = max(1, len(preds))
    r1 = sum(_ngram_f(_toks(p), _toks(r), 1) for p, r in zip(preds, refs)) / n
    r2 = sum(_ngram_f(_toks(p), _toks(r), 2) for p, r in zip(preds, refs)) / n
    rl = sum(_lcs_f(_toks(p), _toks(r)) for p, r in zip(preds, refs)) / n
    lp = sum(len(p.split()) for p in preds) / n
    lr = sum(len(r.split()) for r in refs) / n
    return {"rouge1": r1, "rouge2": r2, "rougeL": rl, "gen_words": lp, "len_ratio": lp / max(1e-9, lr)}


class CosineScorer:
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)

    def __call__(self, preds, refs):
        a = self.m.encode([p or "." for p in preds], convert_to_tensor=True, normalize_embeddings=True)
        b = self.m.encode(refs, convert_to_tensor=True, normalize_embeddings=True)
        return float((a * b).sum(-1).mean())


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def clean_output(text):
    text = text.lstrip("\n ")                       # generated separator (no-template mode)
    text = text.split("\nDialogue:")[0].split("\n\n")[0].split("\n")[0]
    return re.sub(r"\s+", " ", text).strip()


@torch.no_grad()
def generate(model, tok, tpl, texts, gen, batch_size=16):
    model.eval()
    device = next(model.parameters()).device
    pad = tok.pad_token_id
    out = []
    for i in range(0, len(texts), batch_size):
        prompts = [tpl.prompt_ids(t)[0] for t in texts[i:i + batch_size]]
        n = max(len(p) for p in prompts)
        ids = torch.full((len(prompts), n), pad, dtype=torch.long)       # LEFT padding for generation
        att = torch.zeros((len(prompts), n), dtype=torch.long)
        for j, p in enumerate(prompts):
            ids[j, n - len(p):] = torch.tensor(p)
            att[j, n - len(p):] = 1
        res = model.generate(input_ids=ids.to(device), attention_mask=att.to(device),
                             pad_token_id=pad, eos_token_id=tok.eos_token_id, **gen)
        out += [clean_output(tok.decode(r[n:], skip_special_tokens=True)) for r in res]
    return out


class GenEvalTrainer(Trainer):
    """Adds generation metrics (cosine / ROUGE on a val subset) to every evaluation,
    so they can drive best-checkpoint selection and early stopping."""

    def setup_gen_eval(self, tok, tpl, rows, gen, scorer, batch_size):
        self._g = dict(tok=tok, tpl=tpl, rows=rows, gen=gen, scorer=scorer, bs=batch_size)
        self._gen_metrics = {}

    def evaluate(self, *args, **kwargs):
        g = self._g
        preds = generate(self.model, g["tok"], g["tpl"], [r["input"] for r in g["rows"]], g["gen"], g["bs"])
        refs = [r["target"] for r in g["rows"]]
        m = rouge_scores(preds, refs)
        if g["scorer"]:
            m["cosine"] = g["scorer"](preds, refs)
        self._gen_metrics = {k: round(v, 4) for k, v in m.items()}
        self.model.train()
        return super().evaluate(*args, **kwargs)


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
def make_training_args(**kw):
    import inspect
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "evaluation_strategy" in params:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    dropped = [k for k in kw if k not in params]
    if dropped:
        print(f"[info] TrainingArguments not supported by this transformers version, ignored: {dropped}")
    return TrainingArguments(**{k: v for k, v in kw.items() if k in params})


def train(args):
    set_seed(args.seed)
    if args.target_prefix in ("newline", "\\n"):
        args.target_prefix = "\n"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.config.pad_token_id = tok.pad_token_id

    tr_rows = read_jsonl(args.train, need_target=True)
    va_rows = read_jsonl(args.val, need_target=True)
    overlap = {r.get("movie") for r in tr_rows} & {r.get("movie") for r in va_rows} - {None}
    if overlap:
        print(f"[WARN] {len(overlap)} movies in BOTH train and val (leakage): {sorted(overlap)[:5]}")

    # dialogue budget = max_len - template - longest allowed summary
    tpl0 = Template(tok, args.template, 10 ** 9)
    max_input_tokens = args.max_len - len(tpl0.bos + tpl0.pre_ids + tpl0.suf_ids) - args.max_target_tokens - 1
    tpl = Template(tok, args.template, max_input_tokens)

    # drop over-long targets (should not happen with 2-3 sentence labels)
    keep = lambda r: len(tok(args.target_prefix + r["target"], add_special_tokens=False)["input_ids"]) \
        < args.max_target_tokens
    n0 = len(tr_rows)
    tr_rows = [r for r in tr_rows if keep(r)]
    va_rows = [r for r in va_rows if keep(r)]
    train_ds = SFTDataset(tr_rows, tok, tpl, args.target_prefix)
    val_ds = SFTDataset(va_rows, tok, tpl, args.target_prefix)
    lens = [len(x["input_ids"]) for x in train_ds.items]
    print(f"template prefix={tpl.prefix!r} suffix={tpl.suffix!r} | dialogue budget {max_input_tokens} tokens")
    print(f"train: {len(train_ds)} ({n0 - len(tr_rows)} dropped: target > {args.max_target_tokens} tokens; "
          f"{train_ds.truncated} dialogues truncated) | val: {len(val_ds)}")
    print(f"sequence tokens: mean {np.mean(lens):.0f}, p95 {np.percentile(lens, 95):.0f}, max {max(lens)}")

    gen = dict(DEFAULT_GEN, max_new_tokens=args.gen_max_new_tokens)
    scorer = None
    if args.select_metric == "cosine":
        try:
            scorer = CosineScorer(args.embed_model)
        except Exception as e:
            print(f"[WARN] cosine scorer unavailable ({e}); selecting best checkpoint by eval loss")
            args.select_metric = "loss"
    rnd = random.Random(args.seed)
    gen_rows = va_rows if len(va_rows) <= args.gen_eval_samples else rnd.sample(va_rows, args.gen_eval_samples)

    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    targs = make_training_args(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.0,
        warmup_steps=max(1, int(0.03 * steps_per_epoch * args.epochs)),
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=args.select_metric,
        greater_is_better=args.select_metric != "loss",
        logging_steps=50,
        bf16=dtype == torch.bfloat16,
        fp16=False,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
        gradient_checkpointing=args.grad_ckpt,
        remove_unused_columns=False,
    )
    trainer_box = {}

    def compute_metrics(_):
        return trainer_box["t"]._gen_metrics           # generation metrics computed in evaluate()

    kw = dict(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
              data_collator=make_collator(tok.pad_token_id), compute_metrics=compute_metrics,
              preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(-1),
              callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)])
    try:
        trainer = GenEvalTrainer(processing_class=tok, **kw)
    except TypeError:
        trainer = GenEvalTrainer(tokenizer=tok, **kw)
    trainer_box["t"] = trainer
    trainer.setup_gen_eval(tok, tpl, gen_rows, gen, scorer, args.batch_size)
    trainer.train()

    # ---- save best + metadata
    best = Path(args.out) / "best"
    meta = {
        "template": args.template, "prefix": tpl.prefix, "suffix": tpl.suffix,
        "max_input_tokens": max_input_tokens, "max_len": args.max_len,
        "target_prefix": args.target_prefix, "stop": "eos",
        "inference": ("pass ONLY the dialogue tokens; the model generates the separator and summary"
                      if args.template.strip() == "{input}" else
                      "build prefix + dialogue + suffix exactly as stored here"),
        "generation": gen, "base_model": args.model, "select_metric": args.select_metric,
        "train_file": str(args.train), "val_file": str(args.val),
    }
    setattr(trainer.model.config, META_KEY, meta)
    trainer.save_model(str(best))
    tok.save_pretrained(str(best))
    json.dump(meta, open(best / "prompt_meta.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(vars(args), open(Path(args.out) / "run_config.json", "w"), indent=2)

    # ---- full validation predictions
    preds = generate(trainer.model, tok, tpl, [r["input"] for r in va_rows], gen, args.batch_size)
    refs = [r["target"] for r in va_rows]
    final = rouge_scores(preds, refs)
    if scorer:
        final["cosine"] = scorer(preds, refs)
    with open(Path(args.out) / "val_predictions.jsonl", "w", encoding="utf-8") as f:
        for r, p in zip(va_rows, preds):
            f.write(json.dumps({"movie": r.get("movie"), "interval": r.get("interval"), "input": r["input"],
                                "teacher": r["target"], "student": p}, ensure_ascii=False) + "\n")
    json.dump({k: round(v, 4) for k, v in final.items()}, open(Path(args.out) / "val_metrics.json", "w"), indent=2)
    print(f"\nBest model -> {best}\nFinal validation: { {k: round(v, 4) for k, v in final.items()} }")


# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------
def predict(args):
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    if torch.cuda.is_available():
        model.cuda()
    meta = getattr(model.config, META_KEY, None) or json.load(
        open(Path(args.model) / "prompt_meta.json", encoding="utf-8"))
    tpl = Template(tok, meta["template"], meta["max_input_tokens"])
    print(f"template from model: {meta['template']!r} | dialogue budget {meta['max_input_tokens']} tokens")
    rows = read_jsonl(args.input)
    preds = generate(model, tok, tpl, [r["input"] for r in rows], meta["generation"], args.batch_size)
    with open(args.output, "w", encoding="utf-8") as f:
        for r, p in zip(rows, preds):
            f.write(json.dumps({"movie": r.get("movie"), "interval": r.get("interval"), "input": r["input"],
                                "summary": p, **({"teacher": r["target"]} if r.get("target") else {})},
                               ensure_ascii=False) + "\n")
    if rows and all(r.get("target") for r in rows):
        print({k: round(v, 4) for k, v in rouge_scores(preds, [r["target"] for r in rows]).items()})
    print(f"{len(preds)} predictions -> {args.output}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fine-tune / run SmolLM for subtitle summarization.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", required=True)
    t.add_argument("--val", required=True)
    t.add_argument("--out", default="runs/smollm")
    t.add_argument("--model", default="HuggingFaceTB/SmolLM-135M")
    t.add_argument("--template", default=DEFAULT_TEMPLATE,
                   help="Must contain {input}. Use \"{input}\" for NO template: at inference you pass only "
                        "the dialogue (pair with --target_prefix newline)")
    t.add_argument("--target_prefix", default=" ",
                   help="Text the model generates before the summary. Use 'newline' with --template {input}")
    t.add_argument("--max_len", type=int, default=1024, help="Total tokens: template + dialogue + summary")
    t.add_argument("--max_target_tokens", type=int, default=128)
    t.add_argument("--gen_max_new_tokens", type=int, default=80)
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--batch_size", type=int, default=16)
    t.add_argument("--grad_accum", type=int, default=1)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--patience", type=int, default=2)
    t.add_argument("--select_metric", choices=["cosine", "rougeL", "loss"], default="cosine")
    t.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    t.add_argument("--gen_eval_samples", type=int, default=300, help="Val chunks generated per epoch")
    t.add_argument("--grad_ckpt", action="store_true", help="Gradient checkpointing (less GPU memory)")
    t.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    train(args) if args.cmd == "train" else predict(args)


if __name__ == "__main__":
    main()
