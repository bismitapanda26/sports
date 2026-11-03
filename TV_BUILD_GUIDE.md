# Build and run kvtool on the TV (Falcon)

Goal: run the PC-computed KV (`kv.bin`) on the TV with **no CPU prefill**, and
measure CPU and speed. This is the best case that NPU prefill can give.

The idea is simple: **you already build a llama.cpp app for the TV with GBS.
kvtool is just one more small program built the same way, linked to the same
llama.cpp.** You don't build llama.cpp separately.

---

## Step 1 — Add kvtool to your existing TV project

1. Copy `kvtool.cpp` into your app's source folder (next to your app's `.cpp` files).

2. Open your project's `CMakeLists.txt`. Find the line that builds your app and
   links llama, something like:
   ```cmake
   add_executable(my_llm_app main.cpp ...)
   target_link_libraries(my_llm_app PRIVATE llama ...)
   ```
   Add these lines right below it (use **the same libraries** your app links):
   ```cmake
   add_executable(kvtool kvtool.cpp)
   target_link_libraries(kvtool PRIVATE llama)      # same as your app: llama / ggml ...
   target_compile_definitions(kvtool PRIVATE KVTOOL_FA_BOOL)   # your llama.cpp uses bool flash_attn
   install(TARGETS kvtool RUNTIME DESTINATION bin)
   ```
   If your app uses `include_directories(...)` for llama.cpp headers, kvtool
   gets them too. If it uses `target_include_directories(my_llm_app ...)`, copy
   that line and change the name to `kvtool`.

3. Open your `.spec` file (in `packaging/`). In the `%files` section, add:
   ```
   %{_bindir}/kvtool
   ```
   (If your app installs to a different folder, e.g. `/usr/apps/...`, use the
   same folder pattern as your app's line.)

## Step 2 — Build with GBS (same command you use today)

```bash
gbs build -A armv7l --include-all
```
The RPM is created in:
```
~/GBS-ROOT/local/repos/<profile>/armv7l/RPMS/
```
(`<profile>` is the name in your `.gbs.conf`.)

If it fails, look at the **first** `error:` line in the build log.
Common fixes:

| Error | Fix |
|---|---|
| `llama.h: No such file` | add the same include directory your app uses (Step 1.2) |
| `undefined reference to llama_...` | link the same libraries as your app (`llama`, `ggml`, ...) |
| `flash_attn_type` / `flash_attn` not a member | remove the `target_compile_definitions` line, or ask me |
| `File not found: /usr/bin/kvtool` | the `%files` path doesn't match the install path |

## Step 3 — Get the binary out of the RPM (on your PC)

```bash
mkdir rpm_out && cd rpm_out
rpm2cpio ~/GBS-ROOT/local/repos/<profile>/armv7l/RPMS/<your-package>.armv7l.rpm | cpio -idmv
find . -name kvtool
```
Check it's an ARM binary: `file ./usr/bin/kvtool` → should say `ARM, EABI5`.

## Step 4 — Copy everything to the TV

```bash
sdb root on
sdb shell mkdir -p /opt/usr/kvtest
sdb push ./usr/bin/kvtool              /opt/usr/kvtest/
sdb push smollm-q4km.gguf              /opt/usr/kvtest/
sdb push tokens.txt                    /opt/usr/kvtest/
sdb push kv.bin                        /opt/usr/kvtest/
sdb shell chmod +x /opt/usr/kvtest/kvtool
```
`/opt/usr` is usually writable on Tizen; if not, use the folder where you run
your app today.

If llama.cpp is built as **shared** libraries (`libllama.so`, `libggml*.so`),
push them too and point to them:
```bash
sdb push ./usr/lib/libllama.so* ./usr/lib/libggml*.so* /opt/usr/kvtest/
# on the TV, before running:
export LD_LIBRARY_PATH=/opt/usr/kvtest:$LD_LIBRARY_PATH
```
(If your app runs on the TV today without this, the libraries are already
installed or statically linked — skip it.)

## Step 5 — Run on the TV

```bash
sdb shell
cd /opt/usr/kvtest
export KVTOOL_THREADS=2      # same as production
export KVTOOL_FA=0           # same flash-attn setting as your app (0 = off, 1 = on)

# A) baseline: normal CPU prefill + decode (also creates the reference blob)
./kvtool ref     smollm-q4km.gguf tokens.txt ref_tv.bin out_ref_tv.txt

# B) check the PC KV against the TV's own KV
./kvtool compare smollm-q4km.gguf tokens.txt ref_tv.bin kv.bin

# C) simulated NPU: load PC KV, NO CPU prefill, decode only
./kvtool inject  smollm-q4km.gguf tokens.txt ref_tv.bin kv.bin out_inj_tv.txt
```

Note: run step A on the TV (don't copy `ref_state.bin` from the PC), so the
blob matches the TV build exactly.

## Step 6 — Read the results

Each run prints lines like:
```
TIMING ref: CPU prefill      wall=5.700 s  cpu=11.2 s  avg_cpu=49.1% of 4 cores
TIMING ref: decode           wall=3.200 s  cpu=7.4 s   avg_cpu=57.8% of 4 cores
TIMING inject: KV load       wall=0.004 s  cpu=0.004 s ...
TIMING inject: decode        wall=3.200 s  cpu=7.4 s   avg_cpu=57.8% of 4 cores
```
(Numbers are only an example.)

| Number | Meaning |
|---|---|
| ref prefill cpu | CPU work that NPU prefill would remove |
| ref prefill + decode | your current total per request |
| inject KV load + decode | best-case total with NPU prefill |
| `1 − (inject total ÷ ref total)` | max CPU saving NPU prefill can give |
| avg_cpu during decode | CPU % that stays even with NPU prefill |

Also check: `out_inj_tv.txt` should equal your PC `out_inj_q4.txt`, and be
as good as `out_ref_tv.txt` against the GT JSON.

Optional: watch live CPU in a second shell: `top -p $(pidof kvtool)`.

Send me the TIMING lines and I'll calculate the expected average CPU with NPU prefill.
