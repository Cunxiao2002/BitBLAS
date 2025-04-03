import tvm
import ladder
from ladder.graph import IRNode, OutputNode, Edge
from ladder.policy import *
from ladder.te_utils import connect_tensor_graph
from tvm import relay
import os.path as osp
from tvm.contrib.target.onnx import to_onnx
from tvm.relay.testing import run_infer_type
from tvm.contrib import graph_executor
import os
from tvm.script import tir as T
from tvm import te
import torch

# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/" + fname

arch = "cuda"
arch = ladder.arch.__getattribute__(arch)()
dtype="float16"

out_dtype="float16"
perf_map = []

def gemm(M, N, K):
    A = te.placeholder((M, K), name='A', dtype='float16')
    B = te.placeholder((N, K), name='B', dtype='float16')

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name='k')

    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k].astype(out_dtype) * B[j, k].astype(out_dtype), axis=[k]),
        name='C'
    )

    return A, B, C

args_0 = gemm(512, 512, 128)
args_1 = gemm(512, 128, 512)

# args = tuple(connect_tensor_graph(args, arg2, {arg2[0]:arg1[-1]}))
# args = tuple(connect_tensor_graph(args, arg3, {arg3[0]:args[-1]}))
# args = tuple(connect_tensor_graph(args, arg4, {arg4[0]:args[-1]}))

input_args = args_0[:-1]
output_args = [args_0[-1]]

node_0 = IRNode([None for _ in input_args], args_0, "matmul_0")
node_0.add_tag("tensorCoreConfig", [0, 1])


input_args = args_1[:-1]
output_args = [args_1[-1]]
node_1 = IRNode([None for _ in input_args], args_1, "matmul_1")
node_1.add_tag("tensorCoreConfig", [0, 1])

edge = Edge(node_0, node_1, 0, 0)
node_0._out_edges.append(edge)
node_1.set_inputs(0, edge)

output_nodes = [OutputNode(node_1)]
policy = TCPolicy(output_nodes, arch)
configs = policy.emit_config(20)
for config in configs:
    print(config)

compile_results = []
cgen = ladder.CodeGenerator()
for config in configs:
    try:
        cpresult = cgen.compile(output_nodes, config, "cuda", kernel_name="Fused")
    except:
        continue
    compile_results.append(cpresult)
ladder.utils.compile_and_load_parallel(compile_results, arch)
best_latency = 10000
best = None
values = []
for cpresult in compile_results:
    print(cpresult.config)
    code = cpresult.code
    if cpresult.lib is None:
        latency = 10000
    else:
        latency = cpresult.profile()
    values.append(latency)
    if latency < best_latency:
        best_latency = latency
        best = cpresult
    print(latency)
with open("best_code.cu", "w+") as f:
    f.write(code)
torch.cuda.cudart().cudaProfilerStart()
best.get_example_outputs()
torch.cuda.cudart().cudaProfilerStop()
print(best.code)
print("top1: {} \ttop10: {}".format(values[0], min(values)))
print("-" * 80, flush=True)
print("best config: {}".format(best.config))
print("best latency: {}".format(best_latency))
