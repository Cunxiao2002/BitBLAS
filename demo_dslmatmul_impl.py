from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags
from bitblas.base.roller.policy import TensorCorePolicy, DefaultPolicy
from bitblas.base.arch import CUDA
from bitblas.base.utils import apply_and_build
import tvm
from tvm import tir


@tvm.script.ir_module
class MatmulNT:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype=in_dtype)
        B = T.match_buffer(b, [N, K], dtype=in_dtype)
        C = T.match_buffer(c, [M, N], dtype=out_dtype)

        for i, j, k in T.grid(M, N, K):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = tvm.tir.const(0, out_dtype)
                C[vi, vj] = C[vi, vj] + A[vi, vk].astype(out_dtype) * B[
                    vj, vk
                ].astype(out_dtype)

ir_module = MatmulNT
func = ir_module["main"]
target = tvm.target.Target("nvidia/nvidia-a100")
arch = CUDA(target)

# Tune with SIMT Cuda Core
policy = DefaultPolicy(func=func, arch=arch)
try:
    tensorized_func, tags = get_tensorized_func_and_tags(func, arch.target)
except Exception:
    tags = None
# Tune with Tensor Core if possible
if tags:
    policy = TensorCorePolicy(func=tensorized_func, arch=arch, tags=tags)

configs = policy.emit_config(topk=20)
'''
[BitBLAS] Evaluation with config  {'block': [64, 64], 'warp': [32, 32], 'rstep': [32], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.032 ms
[BitBLAS] Evaluation with config  {'block': [32, 128], 'warp': [16, 64], 'rstep': [64], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.021 ms
[BitBLAS] Evaluation with config  {'block': [128, 32], 'warp': [64, 16], 'rstep': [64], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.023 ms
[BitBLAS] Evaluation with config  {'block': [32, 32], 'warp': [16, 16], 'rstep': [128], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.023 ms
[BitBLAS] Evaluation with config  {'block': [32, 64], 'warp': [16, 32], 'rstep': [128], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.027 ms
[BitBLAS] Evaluation with config  {'block': [64, 32], 'warp': [32, 16], 'rstep': [128], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.025 ms
[BitBLAS] Evaluation with config  {'block': [64, 128], 'warp': [32, 64], 'rstep': [32], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.023 ms
[BitBLAS] Evaluation with config  {'block': [128, 64], 'warp': [64, 32], 'rstep': [32], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.025 ms
[BitBLAS] Evaluation with config  {'block': [16, 64], 'warp': [16, 16], 'rstep': [128], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.037 ms
[BitBLAS] Evaluation with config  {'block': [64, 16], 'warp': [16, 16], 'rstep': [128], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.037 ms
[BitBLAS] Evaluation with config  {'block': [128, 128], 'warp': [64, 64], 'rstep': [32], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.026 ms
[BitBLAS] Evaluation with config  {'block': [16, 128], 'warp': [16, 32], 'rstep': [64], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.043 ms
[BitBLAS] Evaluation with config  {'block': [128, 16], 'warp': [32, 16], 'rstep': [64], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.042 ms
[BitBLAS] Evaluation with config  {'block': [32, 256], 'warp': [16, 128], 'rstep': [64], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.025 ms
[BitBLAS] Evaluation with config  {'block': [256, 32], 'warp': [128, 16], 'rstep': [64], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.029 ms
[BitBLAS] Evaluation with config  {'block': [64, 256], 'warp': [32, 128], 'rstep': [32], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.028 ms
[BitBLAS] Evaluation with config  {'block': [256, 64], 'warp': [128, 32], 'rstep': [32], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.027 ms
[BitBLAS] Evaluation with config  {'block': [128, 256], 'warp': [64, 128], 'rstep': [32], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.044 ms
[BitBLAS] Evaluation with config  {'block': [256, 128], 'warp': [128, 64], 'rstep': [32], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.040 ms
[BitBLAS] Evaluation with config  {'block': [16, 256], 'warp': [16, 64], 'rstep': [64], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.047 ms
'''