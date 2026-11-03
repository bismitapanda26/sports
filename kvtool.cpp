// kvtool.cpp - Phase 0: prove that externally computed KV can be injected into llama.cpp.
//
// Modes:
//   kvtool tokens  <model.gguf> <prompt.txt> <tokens.txt>
//   kvtool ref     <model.gguf> <tokens.txt> <ref_state.bin> <out_ref.txt>   [max_new]
//   kvtool compare <model.gguf> <tokens.txt> <ref_state.bin> <kv.bin>
//   kvtool inject  <model.gguf> <tokens.txt> <ref_state.bin> <kv.bin> <out_inj.txt> [max_new]
//
// Flow: prompt has N tokens. Tokens 0..N-2 (T = N-1) are "prefill" -> KV.
// Token N-1 is decoded normally by llama.cpp to get the first logits.
//
// The state blob from llama_state_seq_get_data() is: [meta header][KV data].
// The KV data section is located from the END of the blob (its size is fully
// determined by n_layer, T, row sizes), so we never need to parse the meta header:
// we reuse it verbatim from the reference blob (it only encodes cells 0..T-1, seq 0).
//
// KV data section (per llama.cpp state_write_data):
//   u32 v_trans, u32 n_layer
//   per layer: i32 k_type, u64 k_row_bytes, K[T][nk]
//   if !v_trans: per layer: i32 v_type, u64 v_row_bytes, V[T][nv]
//   if  v_trans: per layer: i32 v_type, u32 v_el_bytes, u32 nv, V[nv][T]
// Requires the default f16 KV cache type.

#include "llama.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static void die(const std::string & m) { fprintf(stderr, "ERROR: %s\n", m.c_str()); exit(1); }

static std::vector<uint8_t> read_bin(const char * p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) die(std::string("cannot open ") + p);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static void write_bin(const char * p, const void * d, size_t n) {
    std::ofstream f(p, std::ios::binary);
    if (!f) die(std::string("cannot write ") + p);
    f.write((const char *) d, n);
}
static std::vector<llama_token> read_tokens(const char * p) {
    std::ifstream f(p);
    if (!f) die(std::string("cannot open ") + p);
    std::vector<llama_token> t; int x;
    while (f >> x) t.push_back(x);
    if (t.size() < 2) die("need at least 2 tokens");
    return t;
}

struct Model {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int L = 0, nk = 0, nv = 0;
};

static Model load(const char * path, int n_ctx = 1024) {
    llama_backend_init();
    Model m;
    llama_model_params mp = llama_model_default_params();
    m.model = llama_model_load_from_file(path, mp);
    if (!m.model) die("failed to load model");
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ctx; cp.n_ubatch = n_ctx;
    cp.n_threads = 4; cp.n_threads_batch = 4;
    // Match your TV build: KVTOOL_FA=0 disables flash attention (V cache stored transposed),
    // KVTOOL_FA=1 forces it on. Unset = llama.cpp default. Use the SAME setting for all steps.
    // (Remove these 3 lines if your llama.cpp is too old to have flash_attn_type.)
    if (const char * fa = getenv("KVTOOL_FA"))
        cp.flash_attn_type = atoi(fa) ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    m.ctx = llama_init_from_model(m.model, cp);
    if (!m.ctx) die("failed to create context");
    m.vocab = llama_model_get_vocab(m.model);
    m.L = llama_model_n_layer(m.model);
    const int n_head = llama_model_n_head(m.model);
    const int n_head_kv = llama_model_n_head_kv(m.model);
    const int head_dim = llama_model_n_embd(m.model) / n_head;
    m.nk = m.nv = n_head_kv * head_dim;
    printf("model: n_layer=%d n_head=%d n_head_kv=%d head_dim=%d  kv_row=%d\n",
           m.L, n_head, n_head_kv, head_dim, m.nk);
    return m;
}
static void unload(Model & m) { llama_free(m.ctx); llama_model_free(m.model); llama_backend_free(); }

static void decode(llama_context * ctx, const llama_token * toks, int n, int pos0, bool logits_last) {
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i] = toks[i]; b.pos[i] = pos0 + i;
        b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
        b.logits[i] = (logits_last && i == n - 1);
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b) != 0) die("llama_decode failed");
    llama_batch_free(b);
}

// Decode `last` at position `pos`, then greedy-generate.
static std::string generate(Model & m, llama_token last, int pos, int max_new) {
    decode(m.ctx, &last, 1, pos, true); ++pos;
    const int n_vocab = llama_vocab_n_tokens(m.vocab);
    std::string out;
    for (int i = 0; i < max_new; ++i) {
        const float * lg = llama_get_logits_ith(m.ctx, -1);
        llama_token best = 0;
        for (llama_token t = 1; t < n_vocab; ++t) if (lg[t] > lg[best]) best = t;
        if (llama_vocab_is_eog(m.vocab, best)) break;
        char buf[256];
        int n = llama_token_to_piece(m.vocab, best, buf, sizeof(buf), 0, false);
        if (n > 0) out.append(buf, n);
        decode(m.ctx, &best, 1, pos, true); ++pos;
    }
    return out;
}

struct Layout {
    size_t data_off = 0, hdr = 8, k_layer = 0, v_layer = 0, k_part = 0;
    uint32_t v_trans = 0;
    int L = 0, T = 0, nk = 0, nv = 0;
    size_t k_data(int l) const { return data_off + hdr + l * k_layer + 12; }
    size_t v_data(int l) const { return data_off + hdr + k_part + l * v_layer + 12; }
};

// Data-section header: u32 v_trans, u32 n_layer, and in newer llama.cpp also
// u32 n_rot_k, u32 n_rot_v (KV Hadamard rotation; always 0 for an f16 cache).
// Both variants are tried, so this works with older and newer llama.cpp.
static Layout find_layout(const std::vector<uint8_t> & blob, int L, int T, int nk, int nv) {
    for (size_t hdr : {16, 8}) {
        for (uint32_t vt = 0; vt < 2; ++vt) {
            Layout lo; lo.L = L; lo.T = T; lo.nk = nk; lo.nv = nv; lo.v_trans = vt; lo.hdr = hdr;
            lo.k_layer = 12 + (size_t) T * nk * 2;          // i32 + u64 + data
            lo.v_layer = 12 + (size_t) T * nv * 2;          // (i32+u64) or (i32+u32+u32): both 12 bytes
            lo.k_part = (size_t) L * lo.k_layer;
            const size_t data = hdr + lo.k_part + (size_t) L * lo.v_layer;
            if (data > blob.size()) continue;
            lo.data_off = blob.size() - data;
            uint32_t a, b;
            memcpy(&a, &blob[lo.data_off], 4); memcpy(&b, &blob[lo.data_off + 4], 4);
            if (a != vt || b != (uint32_t) L) continue;
            if (hdr == 16) {
                uint32_t rk, rv;
                memcpy(&rk, &blob[lo.data_off + 8], 4); memcpy(&rv, &blob[lo.data_off + 12], 4);
                if (rk != 0 || rv != 0) die("KV rotation is enabled (quantized KV cache?) - use the default f16 cache");
            }
            bool ok = true;
            for (int l = 0; l < L && ok; ++l) {
                int32_t kt; uint64_t krow;
                memcpy(&kt, &blob[lo.data_off + hdr + l * lo.k_layer], 4);
                memcpy(&krow, &blob[lo.data_off + hdr + l * lo.k_layer + 4], 8);
                ok = (kt == GGML_TYPE_F16 && krow == (uint64_t) nk * 2);
            }
            if (!ok) continue;
            printf("state blob: total=%zu bytes, meta header=%zu bytes, data header=%zu bytes, v_trans=%u\n",
                   blob.size(), lo.data_off, hdr, vt);
            return lo;
        }
    }
    die("could not locate KV data in state blob (format changed, or KV cache not f16)");
    return {};
}

static float h2f(const uint8_t * p) { ggml_fp16_t h; memcpy(&h, p, 2); return ggml_fp16_to_fp32(h); }

int main(int argc, char ** argv) {
    if (argc < 2) die("usage: kvtool tokens|ref|compare|inject ... (see header comment)");
    const std::string mode = argv[1];

    if (mode == "tokens") {
        if (argc < 5) die("kvtool tokens <model> <prompt.txt> <tokens.txt>");
        Model m = load(argv[2]);
        auto txt = read_bin(argv[3]);
        std::string s(txt.begin(), txt.end());
        std::vector<llama_token> t(s.size() + 16);
        int n = llama_tokenize(m.vocab, s.c_str(), (int) s.size(), t.data(), (int) t.size(), true, true);
        if (n < 0) die("tokenize failed");
        t.resize(n);
        std::ofstream f(argv[4]);
        for (auto x : t) f << x << ' ';
        printf("wrote %d tokens to %s\n", n, argv[4]);
        unload(m);
        return 0;
    }

    if (mode == "ref") {
        if (argc < 6) die("kvtool ref <model> <tokens.txt> <ref_state.bin> <out_ref.txt> [max_new]");
        const int max_new = argc > 6 ? atoi(argv[6]) : 256;
        auto toks = read_tokens(argv[3]);
        const int T = (int) toks.size() - 1;
        Model m = load(argv[2]);
        decode(m.ctx, toks.data(), T, 0, false);                  // normal CPU prefill of 0..T-1
        std::vector<uint8_t> blob(llama_state_seq_get_size(m.ctx, 0));
        size_t n = llama_state_seq_get_data(m.ctx, blob.data(), blob.size(), 0);
        if (n == 0) die("llama_state_seq_get_data failed");
        blob.resize(n);
        write_bin(argv[4], blob.data(), blob.size());
        Layout lo = find_layout(blob, m.L, T, m.nk, m.nv);
        printf("meta header first bytes:");
        for (size_t i = 0; i < std::min<size_t>(lo.data_off, 48); ++i) printf(" %02x", blob[i]);
        printf("\n");
        std::string out = generate(m, toks.back(), T, max_new);
        write_bin(argv[5], out.data(), out.size());
        printf("reference output (%zu chars) -> %s\n%s\n", out.size(), argv[5], out.c_str());
        unload(m);
        return 0;
    }

    if (mode == "compare" || mode == "inject") {
        const bool inj = mode == "inject";
        if (argc < (inj ? 7 : 6)) die(inj
            ? "kvtool inject <model> <tokens.txt> <ref_state.bin> <kv.bin> <out_inj.txt> [max_new]"
            : "kvtool compare <model> <tokens.txt> <ref_state.bin> <kv.bin>");
        auto toks = read_tokens(argv[3]);
        const int T = (int) toks.size() - 1;
        Model m = load(argv[2]);
        auto blob = read_bin(argv[4]);
        auto kv = read_bin(argv[5]);
        Layout lo = find_layout(blob, m.L, T, m.nk, m.nv);
        const size_t k_bytes = (size_t) m.L * T * m.nk * 2, v_bytes = (size_t) m.L * T * m.nv * 2;
        if (kv.size() != k_bytes + v_bytes) die("kv.bin size mismatch (same tokens.txt for both steps?)");
        const uint8_t * K = kv.data();
        const uint8_t * V = kv.data() + k_bytes;

        // Per-layer cosine similarity (reference llama.cpp KV vs exported KV).
        double worst_k = 1, worst_v = 1;
        for (int l = 0; l < m.L; ++l) {
            double dk = 0, nk1 = 0, nk2 = 0, dv = 0, nv1 = 0, nv2 = 0;
            for (int t = 0; t < T; ++t) {
                for (int j = 0; j < m.nk; ++j) {
                    float a = h2f(&blob[lo.k_data(l) + ((size_t) t * m.nk + j) * 2]);
                    float b = h2f(K + (((size_t) l * T + t) * m.nk + j) * 2);
                    dk += a * b; nk1 += a * a; nk2 += b * b;
                }
                for (int j = 0; j < m.nv; ++j) {
                    size_t ri = lo.v_trans ? ((size_t) j * T + t) : ((size_t) t * m.nv + j);
                    float a = h2f(&blob[lo.v_data(l) + ri * 2]);
                    float b = h2f(V + (((size_t) l * T + t) * m.nv + j) * 2);
                    dv += a * b; nv1 += a * a; nv2 += b * b;
                }
            }
            double ck = dk / (sqrt(nk1 * nk2) + 1e-12), cv = dv / (sqrt(nv1 * nv2) + 1e-12);
            worst_k = std::min(worst_k, ck); worst_v = std::min(worst_v, cv);
            printf("layer %2d  cos(K)=%.5f  cos(V)=%.5f\n", l, ck, cv);
        }
        printf("WORST  cos(K)=%.5f  cos(V)=%.5f\n", worst_k, worst_v);
        if (worst_v > 0.98 && worst_k < 0.9)
            printf("HINT: V matches but K does not -> K permutation is wrong.\n");
        if (!inj) { unload(m); return 0; }

        // Overwrite the KV data section with the exported KV, keep the meta header.
        for (int l = 0; l < m.L; ++l) {
            memcpy(&blob[lo.k_data(l)], K + (size_t) l * T * m.nk * 2, (size_t) T * m.nk * 2);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < m.nv; ++j) {
                    size_t ri = lo.v_trans ? ((size_t) j * T + t) : ((size_t) t * m.nv + j);
                    memcpy(&blob[lo.v_data(l) + ri * 2], V + (((size_t) l * T + t) * m.nv + j) * 2, 2);
                }
        }
        if (llama_state_seq_set_data(m.ctx, blob.data(), blob.size(), 0) == 0)
            die("llama_state_seq_set_data failed");
        const int max_new = argc > 7 ? atoi(argv[7]) : 256;
        std::string out = generate(m, toks.back(), T, max_new);   // no CPU prefill here
        write_bin(argv[6], out.data(), out.size());
        printf("injected-KV output (%zu chars) -> %s\n%s\n", out.size(), argv[6], out.c_str());
        unload(m);
        return 0;
    }

    die("unknown mode " + mode);
}
