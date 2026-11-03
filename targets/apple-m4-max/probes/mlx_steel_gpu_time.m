#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// MLX steel GEMM 的参数块（与 params.h 的 GEMMParams 布局一致）。
typedef struct {
    int M, N, K;
    int lda, ldb, ldd;
    int tiles_n, tiles_m;
    int64_t batch_stride_a, batch_stride_b, batch_stride_d;
    int swizzle_log;
    int gemm_k_iterations_aligned;
    int batch_ndim;
} GEMMParams;

int main(int argc, char** argv) {
    @autoreleasepool {
        const char* metallib = argv[1];
        const char* filter = argc > 2 ? argv[2] : "steel_gemm";
        int iters = argc > 3 ? atoi(argv[3]) : 300;
        int M = 256, N = 256, K = 128;
        int BM = 64, BN = 64, BK = 16, WM = 1, WN = 2;
        NSError* err = nil;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib]];
        id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
        if (!lib) { fprintf(stderr, "library error: %s\n", err.localizedDescription.UTF8String); return 1; }
        NSArray<NSString*>* names = lib.functionNames;
        printf("candidates (%lu):\n", (unsigned long)names.count);
        NSMutableArray<NSString*>* picked = [NSMutableArray array];
        for (NSString* n in names) {
            if ([n containsString:[NSString stringWithUTF8String:filter]]) { [picked addObject:n]; }
        }
        // 只要 nn + fp16/fp16 + 对齐的普通 GEMM（block masked 不带 outmask/opmask 变体）。
        NSString* chosen = nil;
        for (NSString* n in picked) {
            if ([n containsString:@"_nn_"] && [n containsString:@"float16_float16"] &&
                ![n containsString:@"outmask"] && ![n containsString:@"splitk"]) { chosen = n; break; }
        }
        if (!chosen && picked.count) chosen = picked[0];
        printf("chosen: %s\n", chosen ? chosen.UTF8String : "(none)");
        if (!chosen) { for (NSString* n in picked) printf("  %s\n", n.UTF8String); return 2; }
        // MLX 的 steel_gemm_fused 依赖 function constants 特化：
        // has_batch(10)=false, use_out_source(100)=false, do_axpby(110)=false,
        // align_M(200)=true, align_N(201)=true, align_K(202)=true。
        MTLFunctionConstantValues* fc = [[MTLFunctionConstantValues alloc] init];
        bool no = false, yes = true;
        [fc setConstantValue:&no type:MTLDataTypeBool atIndex:10];
        [fc setConstantValue:&no type:MTLDataTypeBool atIndex:100];
        [fc setConstantValue:&no type:MTLDataTypeBool atIndex:110];
        [fc setConstantValue:&yes type:MTLDataTypeBool atIndex:200];
        [fc setConstantValue:&yes type:MTLDataTypeBool atIndex:201];
        [fc setConstantValue:&yes type:MTLDataTypeBool atIndex:202];
        id<MTLFunction> fn = [lib newFunctionWithName:chosen constantValues:fc error:&err];
        if (!fn) { fprintf(stderr, "specialize error: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fprintf(stderr, "pso error: %s\n", err.localizedDescription.UTF8String); return 1; }

        const int NSETS = 16;
        id<MTLBuffer> Ab[NSETS], Bb[NSETS], Cb[NSETS];
        srand(7);
        for (int s = 0; s < NSETS; ++s) {
            Ab[s] = [dev newBufferWithLength:M*K*2 options:MTLResourceStorageModeShared];
            Bb[s] = [dev newBufferWithLength:K*N*2 options:MTLResourceStorageModeShared];
            Cb[s] = [dev newBufferWithLength:M*N*2 options:MTLResourceStorageModeShared];
            _Float16* a = (_Float16*)Ab[s].contents;
            _Float16* b = (_Float16*)Bb[s].contents;
            for (int i = 0; i < M*K; ++i) a[i] = (_Float16)((rand() % 2000 - 1000) / 1000.0);
            for (int i = 0; i < K*N; ++i) b[i] = (_Float16)((rand() % 2000 - 1000) / 1000.0);
        }
        GEMMParams params;
        memset(&params, 0, sizeof(params));
        params.M = M; params.N = N; params.K = K;
        params.lda = K; params.ldb = N; params.ldd = N;
        params.tiles_n = N / BN; params.tiles_m = M / BM;
        params.swizzle_log = 0;
        params.gemm_k_iterations_aligned = K / BK;
        params.batch_ndim = 1;
        id<MTLBuffer> pb = [dev newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
        MTLSize grid = MTLSizeMake(N / BN, M / BM, 1);
        MTLSize threads = MTLSizeMake(32, WN, WM);
        printf("grid=%llu,%llu  threads=%llu,%llu,%llu  static_tg=%llu\n",
               (unsigned long long)grid.width, (unsigned long long)grid.height,
               (unsigned long long)threads.width, (unsigned long long)threads.height, (unsigned long long)threads.depth,
               (unsigned long long)pso.staticThreadgroupMemoryLength);

        double best = 1e30, total = 0;
        int reps = 12;
        for (int r = 0; r < reps + 3; ++r) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            [enc setComputePipelineState:pso];
            for (int i = 0; i < iters; ++i) {
                int s = i % NSETS;
                if (i > 0 && s == 0) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [enc setBuffer:Ab[s] offset:0 atIndex:0];
                [enc setBuffer:Bb[s] offset:0 atIndex:1];
                [enc setBuffer:Cb[s] offset:0 atIndex:2];
                [enc setBuffer:pb offset:0 atIndex:3];
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:threads];
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) { fprintf(stderr, "cb error: %s\n", cb.error.localizedDescription.UTF8String); return 1; }
            double per = (cb.GPUEndTime - cb.GPUStartTime) / iters * 1e6;
            if (r >= 3) { total += per; if (per < best) best = per; }
            printf("  rep %d gpu_per_iter=%.3f us\n", r, per);
        }
        printf("RESULT gpu_mean=%.3f us gpu_min=%.3f us\n", total / reps, best);
    }
    return 0;
}
