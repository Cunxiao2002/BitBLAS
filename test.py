# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import bitblas.testing
from bitblas.base.arch import auto_infer_current_arch
from typing import List
import bitblas
# from bitblas.base.template import MatmulTemplate
from bitblas.base import TensorCorePolicy
import tvm
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
from base.roller.node import Edge, OutputNode, PrimFuncNode
from bitblas.base.utils import apply_and_build, apply_and_build_single

def test_elementwise_recommend_hints():
    run_elementwise_recommend_hints([1024, 1024], "float16")
    run_elementwise_recommend_hints([1024], "float16")
    run_elementwise_recommend_hints([1024, 1024, 1024], "float16")


def run_matmul_recommend_hints(
    M: int = 1024,
    N: int = 1024,
    K: int = 1024,
    in_dtype: str = "float16",
    out_dtype: str = "float16",
    accum_dtype: str = "float16",
):
    arch = auto_infer_current_arch()
    # carve_template = MatmulTemplate(
    #     M=M,
    #     N=N,
    #     K=K,
    #     in_dtype=in_dtype,
    #     out_dtype=out_dtype,
    #     accum_dtype=accum_dtype,
    # ).with_arch(arch)


    func = carve_template.equivalent_function()
    assert func is not None, "Function is None"

    # hints = carve_template.recommend_hints(topk=20)
    assert len(hints) > 0, "Hints length is not 20"
    hint = hints[0]
    print(hint.rasterization_plan)
    from bitblas.base.roller.rasterization import NoRasterization
    print(isinstance(hint.rasterization_plan, NoRasterization))

# def test_matmul_recommend_hints():
#     run_matmul_recommend_hints(1024, 1024, 1024, "float16", "float16", "float16")
#     run_matmul_recommend_hints(1024, 1024, 1024, "int8", "int32", "int32")
#     run_matmul_recommend_hints(1024, 1024, 1024, "float16", "float32", "float16")

def test_matmul_recommen_hints_with_function():
    @T.prim_func
    def fused_dense_relu(input0: T.Buffer((T.int64(2073600), T.int64(64)), "float16"), param_0: T.Buffer((T.int64(64), T.int64(64)), "float16"), T_relu_intermediate: T.Buffer((T.int64(2073600), T.int64(64)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_matmul_NT_intermediate = T.alloc_buffer((T.int64(2073600), T.int64(64)))
        for i0, i1, k in T.grid(T.int64(2073600), T.int64(64), T.int64(64)):
            with T.block("T_matmul_NT"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(input0[v_i0, v_k], param_0[v_i1, v_k])
                T.writes(T_matmul_NT_intermediate[v_i0, v_i1])
                with T.init():
                    T_matmul_NT_intermediate[v_i0, v_i1] = T.float16(0)
                T_matmul_NT_intermediate[v_i0, v_i1] = T_matmul_NT_intermediate[v_i0, v_i1] + input0[v_i0, v_k] * param_0[v_i1, v_k]
        for ax0, ax1 in T.grid(T.int64(2073600), T.int64(64)):
            with T.block("T_relu"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(T_matmul_NT_intermediate[v_ax0, v_ax1])
                T.writes(T_relu_intermediate[v_ax0, v_ax1])
                T_relu_intermediate[v_ax0, v_ax1] = T.max(T_matmul_NT_intermediate[v_ax0, v_ax1], T.float16(0))
    
    # mod = Module
    # for g_var, func in mod.functions_items():
    #     if isinstance(func, tvm.tir.PrimFunc):
    #         func.with_attr("global_symbol", g_var.name_hint)
    #         dense = func
    arch = auto_infer_current_arch()


    tensorized_func0, tags0 = bitblas.gpu.matmul_analysis.get_tensorized_func_and_tags(fused_dense_relu, arch.target)

    print(f"the tensorized_func0 is \n {tensorized_func0}")

    node0 = PrimFuncNode(tensorized_func0, name="dense0")
    node1 = PrimFuncNode(tensorized_func0, name="dense1")
    edge = Edge(node0, node1, 0, 0)
    node0._out_edges.append(edge)
    node1.set_inputs(0, edge)

    output_nodes = [OutputNode(node1)]
    policy = TensorCorePolicy.from_output_nodes(output_nodes, arch=arch, tags=tags0)
    hints = policy.emit_config(topk=20)

    for config in hints:
        print(config)

    print("------------------------------------------")

    best_0 = None
    best_1 = None

    best_latency = float('inf')

    # for config in hints:
    #     node0_config0 = [config[node0]]
    #     node1_config1 = [config[node1]]

    #     node0_cpresults, node0_best = apply_and_build_single(tensorized_func0, node0_config0, arch)
    #     node1_cpresults, node1_best = apply_and_build_single(tensorized_func0, node1_config1, arch)
    #     from bitblas.base.roller.rasterization import NoRasterization

    #     if node0_best is None or node1_best is None:
    #         print(f"node0_best or node1_best is None")
    #         continue

    #     total_latency = node0_best.latency + node1_best.latency

    #     if total_latency < best_latency:
    #         best_latency = total_latency
    #         best_0 = node0_best
    #         best_1 = node1_best
    from bitblas.base.roller.rasterization import NoRasterization
    config = hints[0][node0]
    print(config.rasterization_plan)
    print(isinstance(config.rasterization_plan, NoRasterization))


if __name__ == "__main__":
    # tilelang.testing.main()
    # run_matmul_recommend_hints(128, 128, 32, "float16", "float16", "float16")
    test_matmul_recommen_hints_with_function()
