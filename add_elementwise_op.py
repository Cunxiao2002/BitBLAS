from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags
from bitblas.base.roller.policy import TensorCorePolicy, DefaultPolicy
from bitblas.base.arch import CUDA, rdna
from bitblas.base.utils import apply_and_build
import tvm
from tvm import tir


@tvm.script.ir_module
class Add:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, N], dtype=in_dtype)
        B = T.match_buffer(b, [M, N], dtype=in_dtype)
        C = T.match_buffer(c, [M, n], dtype=out_dtype)

        for i, j in T.grid(M, N):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                with T.init():
                    C[vi, vj] = tvm.tir.const(0, out_dtype)
                C[vi, vj] = A[vi, vj].astype(out_dtype) + B[vi, vj].astype(out_dtype)

ir_module = Add
func = ir_module["main"]
target = tvm.target.Target("hip")
arch = rdna(target)

# Tune with SIMT Cuda core
policy = DefaultPolicy(func=func, arch=arch)
tags = None

configs = policy.emit_config(topk=20)
print(configs)