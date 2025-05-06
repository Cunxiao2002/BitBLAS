from bitblas.base.arch import auto_infer_current_arch
from bitblas.base.roller.node import PrimFuncNode
from bitblas.base.roller.policy.tensorcore import TensorCorePolicy
import tvm
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
import bitblas
from bitblas.base import fast_tune
from tvm.tir.function import PrimFunc
from bitblas.base.utils import apply_and_build, apply_and_build_single
from bitblas.base.roller import hint
from bitblas.base.roller.node import Edge, OutputNode, PrimFuncNode
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
log_path = os.path.dirname(os.path.abspath(__file__)) + "/progress_test_1/" + fname

count = 0

bitblas.set_log_level("Debug")

def write_code(code, fname):
    fname = str(count) + "." + fname
    count += 1
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    fname = os.path.join(log_path, fname)
    with open(fname, "w") as f:
        f.write(code)

def write_mod(mod, fname):
    py_fname = fname + ".py"
    write_code(mod.script(show_meta=False), py_fname)
    cu_fname = fname + ".cu"
    write_code(mod.astext(show_meta_data=False), cu_fname)


@ir.transform.module_pass(opt_level=0)
class CustomAnnotateTIROpPattern:
    def transform_module(self, mod, ctx):
        @tvm.tir.transform.prim_func_pass(opt_level=0)
        def transform(func, mod, ctx):
            return func.with_attr("op_pattern", 0)  # 将所有 TIR 标记为 kElemWise
        return transform(mod)

class FusionGroup:
    def __init__(self, node_list: List[PrimFuncNode], latency, compile_result=None):
        self.node_list = node_list
        self.compile_result = compile_result
        self.latency = latency


class FusionPipeline:
    def __init__(self, ordered_nodes):
        # self.mod = mod
        self.ordered_nodes = ordered_nodes
        self.arch = auto_infer_current_arch()
    
    @staticmethod
    def tvm_to_torch(tvm_arrays):
        torch_tensors = []
        for tvm_array in tvm_arrays:
            numpy_array = tvm_array.asnumpy()
            torch_tensor = torch.from_numpy(numpy_array)
            if str(tvm_array.device).startswith('cuda'):
                torch_tensor = torch_tensor.cuda()
            torch_tensors.append(torch_tensor)
        return torch_tensors
    
    @staticmethod
    def ref_compute(tensors):
        # torch ref计算，用于验证准确性
        gemm1 = torch.matmul(tensors[0], tensors[1].T)
        relu1 = F.relu(gemm1)
        gemm2 = torch.matmul(relu1, tensors[2].T)
        relu2 = F.relu(gemm2)

        return relu2
    
    def compute_latency_and_verify(self, mod, data_distribution="uniform", num_repeats=3):
        rt_mod = tvm.build(mod, target=self.arch.target)
        device = self.arch.device
        time_evaluator = rt_mod.time_evaluator(
            rt_mod.entry_name, device, number=num_repeats
        )
        func = retrieve_func_from_module(mod)
        profile_tensors = get_dummy_input_arrays(func, device, distribution=data_distribution)
        latency = time_evaluator(*profile_tensors).mean * 1e3
        print(f"fused latency is {latency} ms")
        
        # 验证正确性
        rt_mod(*profile_tensors[:-1], profile_tensors[-1])
        torch_profile_tensors = self.tvm_to_torch(profile_tensors)
        rt_mod_results = torch_profile_tensors[-1]
        torch_result = self.ref_compute(torch_profile_tensors[:-1])
        torch.allclose(torch_result, rt_mod_results, rtol=1e-3, atol=1e-8)

        return latency
    
    def create_new_primfunc_node(self, primfunc, name):
        tensorized_func, tags = bitblas.gpu.matmul_analysis.get_tensorized_func_and_tags(primfunc, self.arch.target)
        return PrimFuncNode(tensorized_func, name=name), tags

    def build_with_configs(self, hints, prepared_nodes):
        best_latency = 100000
        best_config = None
        best_results = [None] * len(prepared_nodes)
        for config in hints:
            current_latency = 0
            current_results = []
            valid_config = True
            for node in prepared_nodes:
                #TODO: 这里需要改回parallel_build   
                _, node_best = apply_and_build_single(node.prim_func, [config[node]], arch=self.arch)
                if node_best is None:
                    valid_config = False
                    break
                current_latency += node_best.latency
                current_results.append(node_best)
            if not valid_config:
                continue
            if current_latency < best_latency:
                best_latency = current_latency
                best_config = config
                best_results = current_results
        return best_config, best_latency, best_results
    
    def create_fuse_policy(self, ordered_nodes):
        prepared_nodes = []
        tags_list = []
        for node in ordered_nodes:
            prepared_node, tags = self.create_new_primfunc_node(node.prim_func, node.name)
            prepared_nodes.append(prepared_node)
            tags["tensorcore_config"] = [0, 1]
            tags_list.append(tags)
        for i in range(len(prepared_nodes) - 1):
            current_node = prepared_nodes[i]
            next_node = prepared_nodes[i + 1]
            edge = Edge(current_node, next_node, 0, 0)
            current_node._out_edges.append(edge)
            next_node.set_inputs(0, edge)
        output_node = OutputNode(prepared_nodes[-1])
        policy = TensorCorePolicy.from_output_nodes([output_node], arch=self.arch, tags=tags_list[-1])
        return policy, prepared_nodes
    
    def create_dataflow(self, func0, func1):
        func0 = func0.with_attr("target", self.arch.target)
        func1 = func1.with_attr("target", self.arch.target)
        mod = tvm.IRModule({func0.attrs["global_symbol"]: func0, func1.attrs["global_symbol"]: func1})

        func0_gvar = mod.get_global_var(func0.attrs["global_symbol"])
        func1_gvar = mod.get_global_var(func1.attrs["global_symbol"])

        func0_input_buffer = func0.buffer_map[func0.params[0]]
        func0_output_buffer = func0.buffer_map[func0.params[-1]]
        func0_param_buffer = [func0.buffer_map[param] for param in func0.params[1:-1]]

        func1_param_buffer = func1.buffer_map[func1.params[1]]
        func1_output_buffer = func1.buffer_map[func1.params[-1]]

        builder = relax.BlockBuilder()
        input_var = relax.Var("input", relax.TensorStructInfo(func0_input_buffer.shape, func0_input_buffer.dtype))
        func0_param_vars = []
        for i, param_buffer in enumerate(func0_param_buffer):
            param_var = relax.Var(
                f"param_func0_{i}",
                relax.TensorStructInfo(param_buffer.shape, param_buffer.dtype)
            )
            func0_param_vars.append(param_var)
        func1_param_var = relax.Var("param_func1", relax.TensorStructInfo(func1_param_buffer.shape, func1_param_buffer.dtype))
        all_params = [input_var] + func0_param_vars + [func1_param_var]

        with builder.function("main", all_params):
            with builder.dataflow():
                func0_args = [input_var] + func0_param_vars
                func0_output = builder.emit(relax.call_tir(
                    func0_gvar,
                    func0_args,
                    relax.TensorStructInfo(func0_output_buffer.shape, func0_output_buffer.dtype)
                ))
                func1_output = builder.emit(relax.call_tir(
                    func1_gvar,
                    [func0_output, func1_param_var],
                    relax.TensorStructInfo(func1_output_buffer.shape, func1_output_buffer.dtype)
                ))
                dataflow_output = builder.emit_output(func1_output)
            builder.emit_func_output(dataflow_output)
        gv = builder.get()
        mod["main"] = gv["main"]
        return mod
    
    def fuse_node(self, func0, func1):
        relax_mod = self.create_dataflow(func0, func1)
        relax_mod = tvm.script.from_source(relax_mod.script())
        relax_mod = relax.transform.LegalizeOps()(relax_mod)
        relax_mod = CustomAnnotateTIROpPattern()(relax_mod)
        relax_mod = relax.transform.FuseOps()(relax_mod)
        relax_mod = relax.transform.FuseTIR()(relax_mod)
        fused_name = "fused_" + str(func0.attrs["global_symbol"]) + '_' + str(func1.attrs["global_symbol"])
        target_func = relax_mod[fused_name]
        target_func = target_func.with_attr({"global_symbol": "Fused"})
        before_fuse_mod = tvm.IRModule({target_func.attrs["global_symbol"]: target_func})
        before_fuse_mod = tvm.script.from_source(before_fuse_mod.script())
        after_fuse_mod = FuseSharedMemory(before_fuse_mod)
        return after_fuse_mod

    def build_fusion_group(self, top_node):
        idx = self.ordered_nodes.index(top_node)
        fusion_group = [top_node]
        current_prepared = None
        global_best_latency = float('inf')
        fused_mod = None
        while (idx + 1) < len(self.ordered_nodes):
            next_node = self.ordered_nodes[idx + 1]
            policy, prepared_nodes = self.create_fuse_policy(fusion_group + [next_node])
            if current_prepared is None:
                current_prepared = prepared_nodes[0]
            hints = policy.emit_config(topk=1)
            best_config, best_no_fuse_latency, best_results = self.build_with_configs(hints, prepared_nodes)
            if best_config is None:
                break
            print(f"the best config is {best_config}")
            if fused_mod is None:
                fused_mod = self.fuse_node(best_results[0].sch.mod["main"], best_results[-1].sch.mod["main"])
            else:
                fused_mod = self.fuse_node(fused_mod["Fused"], best_results[-1].sch.mod["main"])
            fused_latency = self.compute_latency_and_verify(fused_mod)
            if fused_latency < best_no_fuse_latency:
                global_best_latency = fused_latency
                global_best_mod = fused_mod
                fusion_group.append(next_node)
                idx += 1
                current_prepared = prepared_nodes[-1]
            else:
                break

        print(f"the best latency is {global_best_latency}")
        print(f"the fusion group is {fusion_group}")
        return FusionGroup(fusion_group, latency=global_best_latency)


