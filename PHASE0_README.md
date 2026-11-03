# Phase 0 — Prove KV handoff into llama.cpp (PC only, no NPU)

Goal: show that KV computed **outside** llama.cpp (here: PyTorch, later: NPU)
can be loaded into llama.cpp, and that decode then produces the same output.
If this passes, the NPU-prefill + CPU-decode architecture works.

Files: `hf_kv_export.py`, `kvtool.cpp`

**Already tested end to end** on llama.cpp `fac3a89` (Nov 2026) with a
SmolLM-shaped model (9 heads, 3 KV heads, head_dim 64, f32 and f16 GGUF):
cos(K) = cos(V) = 1.00000, identical generated output, with flash attention
on and off. The negative test (K not permuted) correctly gives cos(K) ≈ 0.02.

---

## 0. Prerequisites (Linux PC / x86_64)

- The **fine-tuned HF model folder** (the one you converted to GGUF).
- Your **Q4_K_M GGUF** (production model).
- The **llama.cpp source at the same version you run on the TV** (pin it).
- Python 3.10+ with: `pip install torch transformers numpy`

## 1. Build llama.cpp and kvtool

```bash
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd ..

g++ -O2 -std=c++17 kvtool.cpp \
  -I llama.cpp/include -I llama.cpp/ggml/include \
  -L llama.cpp/build/bin -lllama -lggml -lggml-base \
  -Wl,-rpath,$(pwd)/llama.cpp/build/bin -o kvtool
```
If linking fails, run `ls llama.cpp/build/bin/*.so` and adjust `-L`/`-l` to what's there.

## 2. Make an f16 GGUF for exact validation

Same weights, no model change — only used to validate the layout cleanly
(Q4 weights add their own small differences).

```bash
python llama.cpp/convert_hf_to_gguf.py <hf_model_dir> --outtype f16 --outfile smollm-f16.gguf
```

## 3. Prepare one real prompt

Put the **exact text you feed in production** (with chat template / special
tokens, exactly as your C++ builds it) into `prompt.txt`.

## 4. Run the steps (f16 model first)

Match your TV build's flash-attention setting for **all** steps:
`export KVTOOL_FA=0` (off, V stored transposed) or `export KVTOOL_FA=1` (on).
If your llama.cpp is older and has no `flash_attn_type`, delete those 3 lines in `load()`.

```bash
# a) tokenize with llama.cpp's tokenizer (single source of truth)
./kvtool tokens smollm-f16.gguf prompt.txt tokens.txt

# b) reference: normal llama.cpp prefill + state dump + greedy output
./kvtool ref smollm-f16.gguf tokens.txt ref_state.bin out_ref.txt

# c) export KV from PyTorch in llama.cpp layout
python hf_kv_export.py --model <hf_model_dir> --tokens tokens.txt --out kv.bin

# d) compare per-layer KV
./kvtool compare smollm-f16.gguf tokens.txt ref_state.bin kv.bin

# e) inject the PyTorch KV (no CPU prefill) and generate
./kvtool inject smollm-f16.gguf tokens.txt ref_state.bin kv.bin out_inj.txt

# f) outputs should match
diff out_ref.txt out_inj.txt && echo "MATCH"
```

## 5. Pass criteria

| Check | f16 GGUF | Q4_K_M GGUF |
|---|---|---|
| WORST cos(K), cos(V) in step d | > 0.999 | ~0.97–0.995 is normal |
| `diff` in step f | identical (or ≤ 1–2 trailing token diffs) | same JSON values on most samples |

Then repeat steps a–f with your **Q4_K_M** GGUF, and run 20–50 different
prompts to check JSON accuracy vs GT.

**Negative test (proves the permutation matters):**
```bash
python hf_kv_export.py --model <hf_model_dir> --tokens tokens.txt --out kv_bad.bin --no-permute
./kvtool compare smollm-f16.gguf tokens.txt ref_state.bin kv_bad.bin   # cos(K) should be low
```

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `could not locate KV data in state blob` | llama.cpp version changed the state format; tell me your version/commit (`git log -1`) |
| compile error on `flash_attn_type` | older llama.cpp: delete the `KVTOOL_FA` lines in `load()` |
| `KV cache is not f16` | don't set `type_k`/`type_v`; keep the default f16 cache |
| cos(V) ≈ 1 but cos(K) low | K permutation wrong (check `head_dim` printed by both tools match) |
| both cos(K) and cos(V) low | different checkpoint for HF vs GGUF, or tokens mismatch |
| `kv.bin size mismatch` | re-run step c with the same `tokens.txt` |
| cos fine but outputs differ a lot | check BOS/special tokens: both sides must use `tokens.txt` |

## 7. Note for production (Phase 3)

In production there is no CPU prefill, so no reference blob. You build the
whole blob yourself. Layout observed on llama.cpp `fac3a89` (single sequence,
f16 cache, SmolLM; all little-endian):

```
[8 bytes]  prefix            -> copy verbatim from any reference blob (constant)
u32        n_stream = 1
u32        cell_count = T
T times:   i32 pos (0..T-1), u32 n_seq_id = 1, i32 seq_id = 0
--- data section ---
u32 v_trans, u32 n_layer, u32 n_rot_k = 0, u32 n_rot_v = 0   (older versions: no n_rot_*)
per layer: i32 type=1 (f16), u64 row_bytes, K[T][192] f16
per layer: V block (row-major, or transposed [192][T] when flash attention is off)
```

Meta header size = 16 + 12·T bytes (352 for T = 28 ✓). **Verify this on your
pinned TV version** with `ref` mode for two prompt lengths: the printed sizes
must follow the same formula. If they do, a small C++ function can build the
blob from NPU output with no CPU prefill.
