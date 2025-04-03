from base.arch import auto_infer_current_arch
from base.roller.node import PrimFuncNode
from base.roller.policy.tensorcore import TensorCorePolicy
import tvm
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
import bitblas
from bitblas.base.arch import auto_infer_current_arch
from bitblas.base import fast_tune
from tvm.tir.function import PrimFunc
from bitblas.base.utils import apply_and_build, apply_and_build_single
from base.roller import hint
from base.roller.node import Edge, OutputNode, PrimFuncNode
from collections import deque
import os
from tvm import relax
import numpy as np
from typing import List
from bitblas.base.utils import CompileResult, retrieve_func_from_module, get_dummy_input_arrays
from FuseSharedMemoryPass import FuseSharedMemory
from tvm import ir, transform, relax
import torch
import torch.nn.functional as F

fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# get current file path
log_path = os.path.dirname(os.path.abspath(__file__)) + "/progress/" + fname

count = 0

bitblas.set_log_level("Debug")

arch = auto_infer_current_arch()

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


weights = []
for i in range(7):  # 根据你的代码需要7个权重常量
    if i < 6:
        # 前6个权重是64x64的
        shape = (64, 64)
    else:
        # 最后一个权重是64x3的
        shape = (64, 3)
    # 创建全零权重（或者随机权重，取决于你的需要）
    weights.append(tvm.nd.array(np.zeros(shape, dtype="float32")))

# 创建模拟的metadata
metadata = {"relax.expr.Constant": weights}

@I.ir_module
class Module:
    @T.prim_func
    def fused_dense1_strided_slice(lv11: T.Buffer((T.int64(2073600), T.int64(64)), "float16"), param_0: T.Buffer((T.int64(8), T.int64(64)), "float16"), T_strided_slice_intermediate: T.Buffer((T.int64(2073600), T.int64(3)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_matmul_NT_intermediate = T.alloc_buffer((T.int64(2073600), T.int64(8)), "float16")
        for i0, i1, k in T.grid(T.int64(2073600), T.int64(8), T.int64(64)):
            with T.block("T_matmul_NT"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(lv11[v_i0, v_k], param_0[v_i1, v_k])
                T.writes(T_matmul_NT_intermediate[v_i0, v_i1])
                with T.init():
                    T_matmul_NT_intermediate[v_i0, v_i1] = T.float16(0)
                T_matmul_NT_intermediate[v_i0, v_i1] = T_matmul_NT_intermediate[v_i0, v_i1] + lv11[v_i0, v_k] * param_0[v_i1, v_k]
        for ax0, ax1 in T.grid(T.int64(2073600), T.int64(3)):
            with T.block("T_strided_slice"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(T_matmul_NT_intermediate[v_ax0, v_ax1])
                T.writes(T_strided_slice_intermediate[v_ax0, v_ax1])
                T_strided_slice_intermediate[v_ax0, v_ax1] = T_matmul_NT_intermediate[v_ax0, v_ax1]

    @T.prim_func
    def fused_dense_relu(input0: T.Buffer((T.int64(2073600), T.int64(64)), "float16"), param_0: T.Buffer((T.int64(64), T.int64(64)), "float16"), T_relu_intermediate: T.Buffer((T.int64(2073600), T.int64(64)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_matmul_NT_intermediate = T.alloc_buffer((T.int64(2073600), T.int64(64)), "float16")
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

    @R.function
    def main(input0: R.Tensor((2073600, 64), dtype="float16"), param_0: R.Tensor((64, 64), dtype="float16"), param_1: R.Tensor((64, 64), dtype="float16"), param_2: R.Tensor((64, 64), dtype="float16"), param_3: R.Tensor((64, 64), dtype="float16"), param_4: R.Tensor((64, 64), dtype="float16"), param_5: R.Tensor((64, 64), dtype="float16"), param_6: R.Tensor((64, 3), dtype="float16")) -> R.Tensor((2073600, 3), dtype="float16"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.fused_dense_relu, (input0, param_0), out_sinfo=R.Tensor((2073600, 64), dtype="float16"))
            lv1 = R.call_tir(cls.fused_dense_relu, (lv, param_1), out_sinfo=R.Tensor((2073600, 64), dtype="float16"))
            lv2 = R.call_tir(cls.fused_dense_relu, (lv1, param_2), out_sinfo=R.Tensor((2073600, 64), dtype="float16"))
            lv3 = R.call_tir(cls.fused_dense_relu, (lv2, param_3), out_sinfo=R.Tensor((2073600, 64), dtype="float16"))
            lv4 = R.call_tir(cls.fused_dense_relu, (lv3, param_4), out_sinfo=R.Tensor((2073600, 64), dtype="float16"))
            lv5 = R.call_tir(cls.fused_dense_relu, (lv4, param_5), out_sinfo=R.Tensor((2073600, 64), dtype="float16"))
            gv = R.call_tir(cls.fused_dense1_strided_slice, (lv5, param_6), out_sinfo=R.Tensor((2073600, 3), dtype="float16"))
            R.output(gv)
        return gv




class WelderTunePass:
    def __init__(self):
        self.nodes_queue = deque()
        self.arch = auto_infer_current_arch()
    
    def add_primfunc_node(self, primfunc: tvm.tir.PrimFunc, name: str) -> PrimFuncNode:
        tensorized_func, tags = bitblas.gpu.matmul_analysis.get_tensorized_func_and_tags(primfunc, self.arch.target)

        node = PrimFuncNode(tensorized_func, name)
        self.nodes_queue.append(node)

        if len(self.nodes_queue) >= 2:
            self._connect_last_two_nodes()
        return node
    
    def _connect_last_two_nodes(self):
        if len(self.queue) < 2:
            return None
        
        node0 = self.nodes_queue[-2]
        node1 = self.nodes_queue[-1]

        edge = Edge(node0, node1, 0, 0)
        node0._out_edges.append(edge)
        node1.set_inputs(0, edge)
    
    def _create_policy(self):
        last_node = self.nodes_queue[-1]
        output_nodes = [OutputNode(last_node)]

        policy = TensorCorePolicy.from_output_nodes(output_nodes, arch = self.arch)

        hints = policy.emit_conifg(topk=20)

        return policy, hints
    
    def process_quene(self):
        policy, hints = self._create_policy()

        if not hints:
            self.nodes_queue.popleft()
            return
        
        config = hints[0]




# 测试两个fused_dense_relu能否放在一起来tune
# mod = Module
# for g_var, func in mod.functions_items():
#     if isinstance(func, tvm.tir.PrimFunc):
#         func.with_attr("global_symbol", g_var.name_hint)
#         dense = func

# dense = Module['fused_dense_relu']

# write_sch(best_0.sch, log_path, "fused_dense_relu_0_best")
# write_sch(best_1.sch, log_path, "fused_dense_relu_1_best")

@tvm.relax.expr_functor.visitor
class TileGraphExtractor(relax.PyExprVisitor):
    def __init__(self, mod: tvm.IRModule):
        super().__init__()
        self.mod = mod
        self.ordered_nodes = []
        self.node_map = {}

    def visit_call_(self, call):
        node_inputs = [self.visit_expr(arg) for arg in call.args]
        super().visit_call_(call)

        # 检查 call_tir
        func_ref = call.args[0]
        if isinstance(func_ref, tvm.ir.GlobalVar):
            primfunc = self.mod[func_ref]
            if isinstance(primfunc, tvm.tir.PrimFunc):
                node = PrimFuncNode(primfunc)
                self.ordered_nodes.append(node)
                self.node_map[node] = call
            else:
                raise NotImplementedError("Not a PrimFunc")
        else:
            raise NotImplementedError("func_ref is not GlobalVar")


class FusionGroup:
    def __init__(self, node_list: List[PrimFuncNode], latency,compile_result=None):
        self.node_list = node_list
        self.compile_result = compile_result
        self.latency = latency

mod = Module
extractor = TileGraphExtractor(mod)
extractor.visit_expr(mod["main"])

ordered_nodes = extractor.ordered_nodes
node_map = extractor.node_map #节点名称->relax call的映射


# 输入两个primfunc
# 首先要创建1个dataflow
# 

def create_dataflow(func0, func1, arch):
    func0 = func0.with_attr("target", arch.target)
    func1 = func1.with_attr("target", arch.target)
    mod = tvm.IRModule({func0.attrs["global_symbol"]: func0, func1.attrs["global_symbol"]: func1})
    
    # 调用MakePackedAPI()前需要补充target
    mod = tvm.tir.transform.MakePackedAPI()(mod)
    func0_gvar = mod.get_global_var(func0.attrs["global_symbol"])
    func1_gvar = mod.get_global_var(func1.attrs["global_symbol"])

    func0_input_buffer = func0.buffer_map[func0.params[0]]
    func0_output_buffer = func0.buffer_map[func0.params[-1]]
    func0_param_buffer = func0.buffer_map[func0.params[1]]


    func1_param_buffer = func1.buffer_map[func1.params[1]]
    func1_output_buffer = func1.buffer_map[func1.params[-1]]
    
    # Create a Relax function that composes func0 and func1
    builder = relax.BlockBuilder()
    
    # Get the input parameter
    input_var = relax.Var("input", relax.TensorStructInfo(func0_input_buffer.shape, func0_input_buffer.dtype))

    func0_param_var = relax.Var("param_func0", relax.TensorStructInfo(func0_param_buffer.shape, func0_param_buffer.dtype))

    func1_param_var = relax.Var("param_func1", relax.TensorStructInfo(func1_param_buffer.shape, func1_param_buffer.dtype))
    
    # Define the main function
    with builder.function("main", [input_var, func0_param_var, func1_param_var]):
        with builder.dataflow():        
            # Call func0
            cls = mod
            func0_output = builder.emit(relax.call_tir(
                func0_gvar, 
                [input_var, func0_param_var],
                relax.TensorStructInfo(func0_output_buffer.shape, func0_output_buffer.dtype)
            ))
            
            # Call func1 with the output of func0
            func1_output = builder.emit(relax.call_tir(
                func1_gvar,
                [func0_output, func1_param_var],
                relax.TensorStructInfo(func1_output_buffer.shape, func1_output_buffer.dtype)
            ))

            dataflow_output = builder.emit_output(func1_output)
        
        # Return the output of func1
        builder.emit_func_output(dataflow_output)
    
    # Add the relax function to the existing module
    gv = builder.get()
    mod["main"] = gv["main"]
    return mod

@ir.transform.module_pass(opt_level=0)
class CustomAnnotateTIROpPattern:
    def transform_module(self, mod, ctx):
        @tvm.tir.transform.prim_func_pass(opt_level=0)
        def transform(func, mod, ctx):
            return func.with_attr("op_pattern", 0)  # 将所有 TIR 标记为 kElemWise
        return transform(mod)


def fuse_node(func0, func1, arch) -> tvm.IRModule:
    relax_mod = create_dataflow(func0, func1, arch)
    relax_mod = tvm.script.from_source(relax_mod.script())
    relax_mod = relax.transform.LegalizeOps()(relax_mod)
    relax_mod = CustomAnnotateTIROpPattern()(relax_mod)
    relax_mod = relax.transform.FuseOps()(relax_mod)
    relax_mod = relax.transform.FuseTIR()(relax_mod)


    # 找到FuseTIR以后的primfunc
    fused_name = "fused_" + str(func0.attrs["global_symbol"]) + '_' + str(func1.attrs["global_symbol"])
    target_func = relax_mod[fused_name]
    target_func = target_func.with_attr({"global_symbol": fused_name})
    
    # for func in relax_mod.functions:
    #     if func.name_hint == "fused_" + str(func0.attrs["global_symbol"]) + '_' + str(func1.attrs["global_symbol"]):
    #         target_func = func
    
    target_mod = tvm.IRModule({target_func.attrs["global_symbol"]: target_func})
    
    # 对其使用FuseSharedMemory pass
    target_mod = tvm.script.from_source(target_mod.script())
    target_mod = FuseSharedMemory(target_mod)
    write_mod(target_mod, log_path, "FuseSharedMemory")

    return target_mod

# 将tvm ndarray转变成torch tensor
def tvm_to_torch(tvm_arrays):
    torch_tensors = []
    for tvm_array in tvm_arrays:
        # Convert TVM NDArray to numpy first
        numpy_array = tvm_array.asnumpy()
        # Then convert numpy array to torch tensor
        torch_tensor = torch.from_numpy(numpy_array)
        # If the original array was on CUDA, move the tensor to CUDA as well
        if str(tvm_array.device).startswith('cuda'):
            torch_tensor = torch_tensor.cuda()
        torch_tensors.append(torch_tensor)
    return torch_tensors

# 使用torch计算，验证结果的正确性
def forward_compute(tensors):
    # Assuming the tensors are in the correct order for your computation
    # tensors[0]: input tensor of shape (2073600, 64)
    # tensors[1] and tensors[2]: weight matrices of shape (64, 64)
    
    # First gemm: matrix multiplication between first two tensors
    gemm1 = torch.matmul(tensors[0], tensors[1].T)
    
    # First relu activation
    relu1 = F.relu(gemm1)
    
    # Second gemm: matrix multiplication with third tensor
    gemm2 = torch.matmul(relu1, tensors[2].T)
    
    # Second relu activation
    relu2 = F.relu(gemm2)
    
    return relu2


# 计算该mod的运行时间
def compute_latency(mod, arch, data_distribution="uniform", num_repeats=3):
    rt_mod = tvm.build(mod, target=arch.target)
    device = arch.device
    time_evaluator = rt_mod.time_evaluator(
        rt_mod.entry_name, device, number=num_repeats
    )

    func = retrieve_func_from_module(mod)
    profile_tensors = get_dummy_input_arrays(func, device, distribution=data_distribution)

    latency = time_evaluator(*profile_tensors).mean * 1e3
    print(f"fused latency is {latency} ms")

    # 验证正确性
    # 首先得到rt_mod的计算结果
    rt_mod(*profile_tensors[:-1], profile_tensors[-1])

    # 用torch计算标准的结果
    torch_profile_tensors = tvm_to_torch(profile_tensors)
    rt_mod_results = torch_profile_tensors[-1]
    torch_result = forward_compute(torch_profile_tensors[:-1])

    torch.allclose(torch_result, rt_mod_results, rtol=1e-3, atol=1e-8)


    return latency

    
    

# 从这个top_node开始往后构建fusion_group
# 构建出这个点开始的fusion_group
def build_fusion_group(top_node, arch) -> FusionGroup:
    idx = ordered_nodes.index(top_node)
    fusion_group = [top_node]
    current_total_latency = 0

    current_fused_func, tags_current = bitblas.gpu.matmul_analysis.get_tensorized_func_and_tags(top_node.prim_func, arch.target)
    current_prepared = PrimFuncNode(current_fused_func, name=top_node.name)

    while(idx + 1) < len(ordered_nodes):
        next_node = ordered_nodes[idx + 1]

        tensorized_next, tags_next = bitblas.gpu.matmul_analysis.get_tensorized_func_and_tags(next_node.prim_func, arch.target)
        next_prepared = PrimFuncNode(tensorized_next, name=next_node.name)

        edge = Edge(current_prepared, next_prepared, 0, 0)
        current_prepared._out_edges.append(edge)
        next_prepared.set_inputs(0, edge)

        output_node = OutputNode(next_prepared)

        tags_next["tensorcore_config"] = [0, 1]
        policy = TensorCorePolicy.from_output_nodes([output_node], arch=arch, tags=tags_next)
        # policy = TensorCorePolicy.from_output_nodes([output_node], arch=arch)
        hints = policy.emit_config(topk=1)

        # 找到最优的config
        best_config = None
        best_no_fuse_latency = 1000000
        global_current_best = None
        global_next_best = None

        for config in hints:
            print(config)
        
        for config in hints:
            _, current_best = apply_and_build_single(current_prepared.prim_func, [config[current_prepared]], arch=arch)
            
            _, next_best = apply_and_build_single(next_prepared.prim_func, [config[next_prepared]], arch=arch)

            if current_best is None or next_best is None:
                continue

            no_fuse_latency = current_best.latency + next_best.latency
            
            if no_fuse_latency < best_no_fuse_latency:
                best_no_fuse_latency = no_fuse_latency
                best_config = config
                global_current_best = current_best
                global_next_best = next_best
            
        if best_config is None:
            break
        else:
            print(f"the best config is {best_config}")
        
        # 进行fuse并判断fuse的结果
        # fused_mod = fuse_node(current_prepared.prim_func, next_prepared.prim_func)
        fused_mod = fuse_node(global_current_best.sch.mod["main"], global_next_best.sch.mod["main"], arch=arch)
        fused_latency = compute_latency(fused_mod, arch)

        if fused_latency < best_no_fuse_latency:
            current_total_latency = fused_latency
            current_fused_mod = fused_mod
            fusion_group.append(next_node)

            idx += 1
            # (TODO-xiao)这里通过current_fused_mod重新构建primfunc有点问题，后续需要重新check
            current_prepared = PrimFuncNode(current_fused_mod.primfunc, name=next_node.name)
        else:
            break

    return FusionGroup(fusion_group, latency=current_total_latency)

        
            
build_fusion_group(ordered_nodes[0], arch=arch)