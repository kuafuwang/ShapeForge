// Harness-owned native batch encoder. No candidate source is compiled into this library.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdint.h>
#include <stdio.h>

typedef struct { uint64_t width, height, depth; } TaicSize;

// Execute one stage whose arguments are all buffers and which needs no fill. The caller retains
// every Objective-C object for the whole call. Encoding, commit and waiting all stay in this
// function; Python's outer wall clock covers plan preparation and the FFI call as well.
int taic_metal_run_batch(
    const void* queue_ptr,
    const void* pso_ptr,
    const void* const* buffers,
    const uint64_t* offsets,
    const uint64_t* slots,
    uint64_t argument_count,
    uint64_t nsets,
    uint64_t iters,
    TaicSize grid,
    TaicSize threads,
    int concurrent,
    int threadgroup_index,
    uint64_t threadgroup_bytes,
    double* gpu_s,
    char* error,
    uint64_t error_capacity
) {
    @autoreleasepool {
        if (!queue_ptr || !pso_ptr || !buffers || !offsets || !slots || !gpu_s ||
            argument_count == 0 || argument_count > 31 || nsets == 0 || iters == 0) {
            return 2;
        }
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queue_ptr;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pso_ptr;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = concurrent
            ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
            : [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        if (threadgroup_index >= 0) {
            [enc setThreadgroupMemoryLength:(NSUInteger)threadgroup_bytes
                                    atIndex:(NSUInteger)threadgroup_index];
        }
        MTLSize grid_size = MTLSizeMake(grid.width, grid.height, grid.depth);
        MTLSize thread_size = MTLSizeMake(threads.width, threads.height, threads.depth);
        for (uint64_t i = 0; i < iters; ++i) {
            uint64_t set = i % nsets;
            if (concurrent && i > 0 && set == 0) {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            for (uint64_t argument = 0; argument < argument_count; ++argument) {
                uint64_t index = set * argument_count + argument;
                id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffers[index];
                [enc setBuffer:buffer
                        offset:(NSUInteger)offsets[index]
                       atIndex:(NSUInteger)slots[argument]];
            }
            [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:thread_size];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            if (error && error_capacity) {
                const char* text = cb.error ? cb.error.localizedDescription.UTF8String : "unknown Metal error";
                snprintf(error, (size_t)error_capacity, "%s", text);
            }
            return 3;
        }
        *gpu_s = (double)(cb.GPUEndTime - cb.GPUStartTime);
        return 0;
    }
}
