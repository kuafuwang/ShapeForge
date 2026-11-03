// GEMM fp16 small: C[256,256] = A[256,128] @ B[128,256]，fp32 累加、输出一次舍入。
// 结构照 MLX steel 的 BlockMMA（BM=64,BN=64,BK=16,WM=1,WN=2）：
//   每线程组 2 个 SIMD 组（threads=(32,2,1)），每个 SG 负责 64(M)x32(N)。
//   每个 K 步（8）：A 8 fragment（M=64,K=8），B 4 fragment（K=8,N=32），C 32 fragment。
//   操作数按 BaseMMAFrag::load 的布局直接从 device 内存读进寄存器，不经 threadgroup 中转。
/* taic-launch
launch gemm_kernel grid=8,16,1 threads=32,1,1
*/
kernel void gemm_kernel(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]])
{
    constexpr int M = 256, N = 256, K = 128;
    constexpr int BM = 16, BN = 32, BK = 16;
    constexpr int LDA = K, LDB = N, LDD = N;

    const int c_row = tid.y * BM;
    const int c_col = tid.x * BN;

    const short qid = simd_lane_id / 4;
    const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
    const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

    const short tm = 8 * (simd_group_id / 2);
    const short tn = 8 * (simd_group_id % 2);

    // 与 init 相同的 fragment 坐标：A 基址 = (m0)*LDA + k，B 基址 = k*LDB + n0。
    device const half* Ap = A + (c_row + tm) * LDA;
    device const half* Bp = B + (c_col + tn);

    simdgroup_float8x8 acc[2][4];
    for (short i = 0; i < 2; ++i)
        for (short j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

    for (int k = 0; k < K; k += 8) {
        simdgroup_half8x8 a[2], b[4];
        for (short i = 0; i < 2; ++i)
            simdgroup_load(a[i], Ap + (i * 8) * LDA + k, LDA);
        for (short j = 0; j < 4; ++j)
            simdgroup_load(b[j], Bp + k * LDB + (j * 8), LDB);
        for (short i = 0; i < 2; ++i)
            for (short j = 0; j < 4; ++j)
                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }

    // 与 init 相同的 fragment 写回坐标：行 = (q&4)+((lane/2)%4)，列 = (q&2)*2+(lane%2)*2。
    const short q = simd_lane_id / 4;
    const short row = (q & 4) + ((simd_lane_id / 2) % 4);
    const short col = (q & 2) * 2 + (simd_lane_id % 2) * 2;
    device half* Dp = C + (c_row + tm) * LDD + (c_col + tn);
    for (short i = 0; i < 2; ++i) {
        for (short j = 0; j < 4; ++j) {
            auto e = acc[i][j].thread_elements();
            device half* dst = Dp + (i * 8 + row) * LDD + (j * 8 + col);
            dst[0] = half(e[0]);
            dst[1] = half(e[1]);
        }
    }
}
