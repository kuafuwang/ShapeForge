"""Re-runnable, non-publishable GEMM Split-K launch-floor probe."""
import json
import statistics
from pathlib import Path
import numpy as np
from taic.task_binding import bind_task
from tcenv.backends.metal.launch import parse_launch_spec
from tcenv.backends.metal.native_batch import prepare_native_batch
from tcenv.backends.metal.runtime import get_context
from tcenv.backends.metal.stage_binding import bind_pipeline_arguments
from tcenv.backends.metal.track_adapter import msl_single_geometry_resolver
from tcenv.backends.metal.v2_runtime import V2ArgSet, reflection_for_kernel
from tcenv.gpulock import gpu_lock
from tcenv.target import load_target
from tcenv.track import abi_limits_from_target, candidate_from_single_source, derive_tensor_memory_contract, resolve_geometry



def run_probe(output: Path) -> dict:
    binding = bind_task('gemm', 'fp16', 'small', target_id='apple-m4-max')
    contract = binding.locator.load()
    symbols = dict(contract.instance.symbols)
    target = load_target('apple-m4-max')
    abi = derive_tensor_memory_contract(contract.instance, target=target)
    limits = abi_limits_from_target(target)
    header = '#include <metal_stdlib>\nusing namespace metal;\n' + ''.join(f'constant constexpr uint {k} = {v}u;\n' for k, v in symbols.items())
    SOURCES = {
        'single_stage': '''
    kernel void probe_single(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]], device half* C [[buffer(2)]], uint lid [[thread_index_in_threadgroup]]) {
      if (lid == 0u) C[0] = half(0.0f);
    }
    /* taic-launch
    math fast
    launch probe_single grid=1,1,1 threads=32,1,1
    */
    ''',
        'two_stage_no_fill': '''
    kernel void probe_producer(device const half* A [[buffer(0)]], device float* partial [[buffer(1)]], uint lid [[thread_index_in_threadgroup]]) {
      if (lid == 0u) partial[0] = 0.0f;
    }
    kernel void probe_consumer(device const half* A [[buffer(0)]], device const float* partial [[buffer(1)]], device half* C [[buffer(2)]], device const half* B [[buffer(3)]], uint lid [[thread_index_in_threadgroup]]) {
      if (lid == 0u) C[0] = half(partial[0]);
    }
    /* taic-launch
    math fast
    scratch partial fp32 M*N zero
    launch probe_producer grid=1,1,1 threads=32,1,1 scratch=partial:write
    launch probe_consumer grid=1,1,1 threads=32,1,1 scratch=partial:read
    */
    ''',
    }
    ctx = get_context()
    prepare_native_batch(ctx)
    result = {
        'schema_id': 'shapeforge.gemm-splitk-launch-floor-probe.1',
        'not_a_candidate': True,
        'correctness': 'intentionally incomplete launch-floor kernels; not evaluated and not publishable',
        'publication_allowed': False,
        'target_id': 'apple-m4-max',
        'size': 'small',
        'baseline_us': {'mlx': 2.2444435181820985, 'torch_mps': 13.602171549552798},
        'baseline_source': 'tests/support/golden/shared/shape-portfolio-阶段验收/P3/evidence/gemm-performance.json',
        'existing_candidate_us': 1.8763327234945097,
        'arms': {},
    }
    baseline = result['baseline_us']['mlx']
    with gpu_lock('gemm-splitk-launch-floor'):
        for label, source in SOURCES.items():
            spec = parse_launch_spec(source, symbols)
            resolved = resolve_geometry(msl_single_geometry_resolver(contract)(candidate_from_single_source(source)), instance=contract.instance, contract=abi, limits=limits)
            kernels = list(ctx.compile_all(header + source, [stage.kernel for stage in spec.stages], math=spec.math))
            reflection = {kernel.name: tuple(reflection_for_kernel(kernel)) for kernel in kernels}
            bindings = bind_pipeline_arguments(resolved, contract=abi, reflection=reflection, limits=limits)
            sets = [V2ArgSet(ctx, abi, resolved) for _ in range(6)]
            for one in sets:
                one.buffers['A'].write(np.ones(symbols['M'] * symbols['K'], dtype=np.float16).tobytes())
                one.buffers['B'].write(np.ones(symbols['K'] * symbols['N'], dtype=np.float16).tobytes())
            args = [[one.stage_args(binding) for binding in bindings] for one in sets]
            fills = [[] for _ in sets]
            grids = [stage.grid for stage in spec.stages]
            threads = [stage.threads for stage in spec.stages]
            tg = [0 for _ in spec.stages]
            ctx.run_pipeline(kernels, grids, threads, tg, args, fills, iters=50, dispatch='concurrent')
            walls, gpus = [], []
            for _ in range(9):
                timing = ctx.run_pipeline(kernels, grids, threads, tg, args, fills, iters=200, dispatch='concurrent')
                walls.append(float(timing.wall_s) / 200 * 1e6)
                gpus.append(float(timing.gpu_s) / 200 * 1e6)
            arm = {
                'wall_us_median': statistics.median(walls), 'gpu_us_median': statistics.median(gpus),
                'wall_samples_us': walls, 'gpu_samples_us': gpus, 'iters_per_sample': 200,
                'rotation_sets': 6, 'dispatch': 'concurrent, one command buffer, native batch when eligible',
                'fills_per_iteration': 0, 'stages': len(kernels),
                'wall_over_existing_candidate': statistics.median(walls) / result['existing_candidate_us'],
            }
            result['arms'][label] = arm
            print(label, json.dumps(arm, indent=2), flush=True)
    result['two_stage_floor_exceeds_existing_candidate'] = result['arms']['two_stage_no_fill']['wall_us_median'] > result['existing_candidate_us']
    result['two_stage_floor_exceeds_mlx_baseline'] = result['arms']['two_stage_no_fill']['wall_us_median'] > baseline
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('SUMMARY', json.dumps({k: result[k] for k in ('existing_candidate_us','baseline_us','two_stage_floor_exceeds_existing_candidate','two_stage_floor_exceeds_mlx_baseline')}, indent=2))
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run_probe(args.output)


if __name__ == "__main__":
    main()
