# Triton specification: Mamba-2 (SSD) chunk scan, element type {{TL}}
# One program handles one (batch, head, chunk) of CHUNK consecutive time steps and produces the output rows of that chunk:
#   y[t] = sum_{s <= t, same chunk} (C[t] . B[s]) * exp(cs[t] - cs[s]) * dt[s] * x[s]      (within the chunk)
#        + exp(cs[t]) * (C[t] . state)                                                      (carried-in state of the chunk)
#        + D * x[t]                                                                         (skip connection)
# Layouts (all contiguous): x, y [BATCH, SEQLEN, NHEADS, HEADDIM]; dt [BATCH, SEQLEN, NHEADS]; dA_cumsum [BATCH, NHEADS, NCHUNKS, CHUNK]
# (inclusive cumulative sum of dt * A inside each chunk, non-increasing); Bmat, Cmat [BATCH, SEQLEN, DSTATE] (one group, shared by all heads);
# Dskip [NHEADS]; states [BATCH, NCHUNKS, NHEADS, HEADDIM, DSTATE] (the SSM state entering each chunk).
# Launch: grid = (NCHUNKS, BATCH * NHEADS), num_warps = {{NW}}, BATCH = {{BATCH}}, SEQLEN = {{SEQLEN}}, NHEADS = {{NHEADS}}, HEADDIM = {{HEADDIM}}, DSTATE = {{DSTATE}}, CHUNK = {{CHUNK}}, NCHUNKS = {{NCHUNKS}}
import triton
import triton.language as tl


@triton.jit
def mamba2_chunk_scan_kernel(x_ptr, dt_ptr, da_cumsum_ptr, bmat_ptr, cmat_ptr, dskip_ptr, states_ptr, y_ptr,
                             BATCH: tl.constexpr, SEQLEN: tl.constexpr, NHEADS: tl.constexpr, HEADDIM: tl.constexpr,
                             DSTATE: tl.constexpr, CHUNK: tl.constexpr, NCHUNKS: tl.constexpr):
    pid_c = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    b = pid_bh // NHEADS
    h = pid_bh % NHEADS
    offs_l = tl.arange(0, CHUNK)
    offs_p = tl.arange(0, HEADDIM)
    offs_n = tl.arange(0, DSTATE)
    t = b * SEQLEN + pid_c * CHUNK + offs_l                                            # global time-step rows of this chunk
    x = tl.load(x_ptr + (t[:, None] * NHEADS + h) * HEADDIM + offs_p[None, :])         # [CHUNK, HEADDIM]
    dt = tl.load(dt_ptr + t * NHEADS + h).to(tl.float32)                              # [CHUNK]
    cs = tl.load(da_cumsum_ptr + ((b * NHEADS + h) * NCHUNKS + pid_c) * CHUNK + offs_l).to(tl.float32)   # [CHUNK]
    bm = tl.load(bmat_ptr + t[:, None] * DSTATE + offs_n[None, :])                     # [CHUNK, DSTATE]
    cm = tl.load(cmat_ptr + t[:, None] * DSTATE + offs_n[None, :])                     # [CHUNK, DSTATE]
    st = tl.load(states_ptr + (((b * NCHUNKS + pid_c) * NHEADS + h) * HEADDIM + offs_p[:, None]) * DSTATE + offs_n[None, :])   # [HEADDIM, DSTATE]
    acc = tl.dot(cm, tl.trans(st)) * tl.exp(cs)[:, None]                               # carried-in state, fp32 [CHUNK, HEADDIM]
    cb = tl.dot(cm, tl.trans(bm))                                                      # C[t] . B[s], fp32 [CHUNK, CHUNK]
    w = tl.where(offs_l[:, None] >= offs_l[None, :], cb * tl.exp(cs[:, None] - cs[None, :]) * dt[None, :], 0.0)
    acc = tl.dot(w.to(x.dtype), x, acc)                                                # intra-chunk term
    acc += tl.load(dskip_ptr + h).to(tl.float32) * x.to(tl.float32)
    tl.store(y_ptr + (t[:, None] * NHEADS + h) * HEADDIM + offs_p[None, :], acc.to({{TL}}))
