from base.roller import hint
from base.roller.node import Edge, OutputNode, PrimFuncNode
import numpy as np
import os
from typing import Dict
import time
import bitblas
from bitblas import tvm as tvm
from tvm import relay, relax, runtime, transform, tir, ir
from tvm.relax.testing import relay_translator, nn
from tvm.target.target import Target
import tvm.relay.testing
from tvm.ir.module import IRModule
from bitblas.relax import ApplyDefaultSchedule, ApplyFastTuning
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
# from bitblas.base.welder.test_relax import OperatorExtractor
from bitblas.base import fast_tune
from bitblas.base.arch import auto_infer_current_arch
from tvm.tir.function import PrimFunc
from bitblas.base.utils import apply_and_build


fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# get current file path
log_path = os.path.dirname(os.path.abspath(__file__)) + "/progress/" + fname

count = 0

bitblas.set_log_level("Debug")

target = tvm.target.Target("cuda")

def write_code(code, path, fname):
    global count
    fname = str(count) + "." + fname
    count += 1
    if not os.path.exists(path):
        os.makedirs(path)
    fname = os.path.join(path, fname)
    with open(fname, "w") as f:
        f.write(code)


def write_sch(sch, path, fname):
    py_fname = fname + ".py"
    write_code(sch.mod["main"].script(), path, py_fname)
    cu_fname = fname + ".cu"
    write_code(sch.mod.astext(), path, cu_fname)


def write_mod(mod, path, fname):
    py_fname = fname + ".py"
    write_code(mod.script(show_meta=False), path, py_fname)
    cu_fname = fname + ".cu"
    write_code(mod.astext(show_meta_data=False), path, cu_fname)



@I.ir_module
class MatmulReluModule:
    @R.function
    def main(
        x: R.Tensor((128, 128), "float32"),
        w: R.Tensor((128, 128), "float32")
    ) -> R.Tensor((128, 128), "float32"):
        with R.dataflow():
            # Matrix multiplication
            lv0 = R.matmul(x, w)
            # ReLU activation
            lv1 = R.nn.relu(lv0)
            # Mark output
            R.output(lv1)
        return lv1

# 创建一个将op_pattern统一的pass
@ir.transform.module_pass(opt_level=0)
class CustomAnnotateTIROpPattern:
    def transform_module(self, mod, ctx):
        @tvm.tir.transform.prim_func_pass(opt_level=0)
        def transform(func, mod, ctx):
            return func.with_attr("op_pattern", 0)  # 将所有 TIR 标记为 kElemWise
        return transform(mod)

# get the fused config
def create_gemm(M, N, K):
    @T.prim_func
    def gemm(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [M, K], dtype="float16")
        B = T.match_buffer(b, [K, N], dtype="float16")
        C = T.match_buffer(c, [M, N], dtype="float16")

        for i, j, k in T.grid(M, N, K):
            with T.block("gemm"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float16(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    return gemm


arch = auto_infer_current_arch()
gemm_0 = create_gemm(512, 512, 128)
gemm_1 = create_gemm(512, 128, 512)

tensorized_func0, tags0 = bitblas.gpu.matmul_analysis.get_tensorized_func_and_tags(gemm_0, arch.target)
# print(f"tags0 is {tags0}")
tensorized_func1, tags1 = bitblas.gpu.matmul_analysis.get_tensorized_func_and_tags(gemm_1, arch.target)
# print(f"tags1 is {tags1}")

node0 = PrimFuncNode(tensorized_func0, name="matmul_0")
node1 = PrimFuncNode(tensorized_func1, name="matmul_1")

edge = Edge(node0, node1, 0, 0)
node0._out_edges.append(edge)
node1.set_inputs(0, edge)

output_nodes = [OutputNode(node1)]
policy = bitblas.base.policy.TensorCorePolicy.from_output_nodes(output_nodes, arch=arch, tags=tags1)

hints = policy.emit_config(topk=20)


for config in hints:
    print(config)

# 根据config进行schedule
# 在relax中fuse
# codegen
node0_configs = [config[node0] for config in hints]
node1_configs = [config[node1] for config in hints]

# node0_cpresults, node0_best = apply_and_build(gemm_0, node0_configs, arch, parallel_build=False)
# node1_cpresults, node1_best = apply_and_build(gemm_1, node1_configs, arch, parallel_build=False)



# print(node0_best.sch.mod.script())
# write_sch(node0_best.sch, log_path, "node0_best")
# write_sch(node1_best.sch, log_path, "node1_best")



