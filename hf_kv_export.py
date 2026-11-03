#!/usr/bin/env python3
"""
Phase 0 - Step 2: Export KV cache from the HF (PyTorch) model in llama.cpp layout.

Reads the token IDs produced by llama.cpp (tokens.txt), runs tokens[0 .. N-2]
through the fine-tuned HF model, and writes kv.bin:

    K : float16 [n_layer][T][n_kv_head * head_dim]   (llama.cpp RoPE order)
    V : float16 [n_layer][T][n_kv_head * head_dim]
    with T = N - 1

K is permuted from HF "rotate_half" order to llama.cpp interleaved order,
matching the Q/K weight permutation done by convert_hf_to_gguf.py.

Usage:
    python hf_kv_export.py --model <hf_model_dir> --tokens tokens.txt --out kv.bin
    (add --no-permute to test the un-permuted K; it should FAIL the compare step)
"""
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM


def get_layer_kv(pkv, i):
    """Return (K, V) for layer i across transformers versions. Shape [1, n_kv, T, hd]."""
    if hasattr(pkv, "layers"):                 # newer Cache API
        return pkv.layers[i].keys, pkv.layers[i].values
    if hasattr(pkv, "key_cache"):              # DynamicCache (older)
        return pkv.key_cache[i], pkv.value_cache[i]
    return pkv[i][0], pkv[i][1]                # legacy tuple


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="fine-tuned HF model dir (same one converted to GGUF)")
    ap.add_argument("--tokens", required=True, help="tokens.txt written by kvtool tokens")
    ap.add_argument("--out", default="kv.bin")
    ap.add_argument("--no-permute", action="store_true", help="skip K permutation (negative test)")
    args = ap.parse_args()

    ids = [int(x) for x in open(args.tokens).read().split()]
    if len(ids) < 2:
        raise SystemExit("need at least 2 tokens")
    prefill = ids[:-1]                          # last token is decoded by llama.cpp
    T = len(prefill)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager")
    model.eval()
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_head)
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_head
    print(f"model: layers={n_layer} heads={n_head} kv_heads={n_kv} head_dim={hd}  T={T}")

    with torch.no_grad():
        out = model(torch.tensor([prefill]), use_cache=True)
    pkv = out.past_key_values

    K_all, V_all = [], []
    for i in range(n_layer):
        k, v = get_layer_kv(pkv, i)
        k = k[0].permute(1, 0, 2).float().numpy()   # [T, n_kv, hd]
        v = v[0].permute(1, 0, 2).float().numpy()   # [T, n_kv, hd]
        if not args.no_permute:
            # HF pairs (j, j+hd/2) -> llama.cpp pairs (2j, 2j+1)
            k = k.reshape(T, n_kv, 2, hd // 2).swapaxes(-1, -2).reshape(T, n_kv, hd)
        K_all.append(k.reshape(T, n_kv * hd))
        V_all.append(v.reshape(T, n_kv * hd))

    K = np.stack(K_all).astype(np.float16)          # [L, T, nk]
    V = np.stack(V_all).astype(np.float16)          # [L, T, nv]
    with open(args.out, "wb") as f:
        K.tofile(f)
        V.tofile(f)
    print(f"wrote {args.out}: K{K.shape} V{V.shape} float16, "
          f"{K.nbytes + V.nbytes} bytes, permute={'off' if args.no_permute else 'on'}")


if __name__ == "__main__":
    main()
