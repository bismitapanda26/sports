// kv_inject.h - drop-in KV injection for your existing llama.cpp app.
//
// Uses your app's own model/context (same settings that already work on the TV).
// Two functions:
//   kv_save_ref(ctx, path)                      -> after CPU prefill of N-1 tokens, save state blob
//   kv_inject(ctx, model, T, kv_path, ref_path) -> load PC/NPU KV (kv.bin) instead of CPU prefill
//
// kv.bin layout (from hf_kv_export.py / NPU): f16 K[n_layer][T][kv_row], then V[n_layer][T][kv_row]
// ref blob must come from THIS app on THIS TV with the SAME model and SAME T (same tokens).
// Requires default f16 KV cache. Header-only: just #include "kv_inject.h" in one .cpp file.

#pragma once
#include "llama.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

namespace kvi {

static bool read_file(const char * p, std::vector<uint8_t> & out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "[kv] cannot open %s\n", p); return false; }
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

struct Layout {
    size_t data_off = 0, hdr = 8, k_layer = 0, v_layer = 0, k_part = 0;
    uint32_t v_trans = 0;
    size_t k_data(int l) const { return data_off + hdr + l * k_layer + 12; }
    size_t v_data(int l) const { return data_off + hdr + k_part + l * v_layer + 12; }
};

// Locate the KV data section from the END of the blob (works for old and new llama.cpp).
static bool find_layout(const std::vector<uint8_t> & b, int L, int T, int nk, int nv, Layout & lo) {
    for (size_t hdr : {16, 8}) {
        for (uint32_t vt = 0; vt < 2; ++vt) {
            Layout x; x.hdr = hdr; x.v_trans = vt;
            x.k_layer = 12 + (size_t) T * nk * 2;
            x.v_layer = 12 + (size_t) T * nv * 2;
            x.k_part  = (size_t) L * x.k_layer;
            const size_t data = hdr + x.k_part + (size_t) L * x.v_layer;
            if (data > b.size()) continue;
            x.data_off = b.size() - data;
            uint32_t a, n;
            memcpy(&a, &b[x.data_off], 4); memcpy(&n, &b[x.data_off + 4], 4);
            if (a != vt || n != (uint32_t) L) continue;
            bool ok = true;
            for (int l = 0; l < L && ok; ++l) {
                int32_t t; uint64_t row;
                memcpy(&t,   &b[x.data_off + hdr + l * x.k_layer], 4);
                memcpy(&row, &b[x.data_off + hdr + l * x.k_layer + 4], 8);
                ok = (t == GGML_TYPE_F16 && row == (uint64_t) nk * 2);
            }
            if (!ok) continue;
            lo = x;
            return true;
        }
    }
    fprintf(stderr, "[kv] KV data not found in ref blob (different T, model, or non-f16 cache?)\n");
    return false;
}

// Call right after CPU prefill of tokens 0..T-1 (before decoding the last prompt token).
static bool kv_save_ref(llama_context * ctx, const char * path) {
    std::vector<uint8_t> blob(llama_state_seq_get_size(ctx, 0));
    const size_t n = llama_state_seq_get_data(ctx, blob.data(), blob.size(), 0);
    if (n == 0) { fprintf(stderr, "[kv] state_seq_get_data failed\n"); return false; }
    std::ofstream f(path, std::ios::binary);
    f.write((const char *) blob.data(), n);
    fprintf(stderr, "[kv] saved ref blob %s (%zu bytes)\n", path, n);
    return true;
}

// Replaces CPU prefill: fills KV for tokens 0..T-1 from kv_path. KV cache must be empty.
// After this, decode the last prompt token at position T (with logits) as usual.
static bool kv_inject(llama_context * ctx, const llama_model * model, int T,
                      const char * kv_path, const char * ref_path) {
    const int L  = llama_model_n_layer(model);
    const int nh = llama_model_n_head(model);
    const int nk = llama_model_n_head_kv(model) * (llama_model_n_embd(model) / nh);
    const int nv = nk;

    std::vector<uint8_t> blob, kv;
    if (!read_file(ref_path, blob) || !read_file(kv_path, kv)) return false;

    const size_t kb = (size_t) L * T * nk * 2, vb = (size_t) L * T * nv * 2;
    if (kv.size() != kb + vb) {
        fprintf(stderr, "[kv] kv.bin is %zu bytes, expected %zu (T=%d L=%d row=%d) - same tokens?\n",
                kv.size(), kb + vb, T, L, nk);
        return false;
    }
    Layout lo;
    if (!find_layout(blob, L, T, nk, nv, lo)) return false;

    const uint8_t * K = kv.data();
    const uint8_t * V = kv.data() + kb;
    for (int l = 0; l < L; ++l) {
        memcpy(&blob[lo.k_data(l)], K + (size_t) l * T * nk * 2, (size_t) T * nk * 2);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < nv; ++j) {
                const size_t ri = lo.v_trans ? ((size_t) j * T + t) : ((size_t) t * nv + j);
                memcpy(&blob[lo.v_data(l) + ri * 2], V + (((size_t) l * T + t) * nv + j) * 2, 2);
            }
    }
    if (llama_state_seq_set_data(ctx, blob.data(), blob.size(), 0) == 0) {
        fprintf(stderr, "[kv] state_seq_set_data failed\n");
        return false;
    }
    fprintf(stderr, "[kv] injected KV: T=%d layers=%d v_trans=%u\n", T, L, lo.v_trans);
    return true;
}

} // namespace kvi
