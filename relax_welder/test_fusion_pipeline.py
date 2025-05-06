from fusion_pipeline import FusionPipeline
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
import tvm
from tvm import relax
from bitblas.base.roller.node import PrimFuncNode

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
            # gv = R.call_tir(cls.fused_dense1_strided_slice, (lv5, param_6), out_sinfo=R.Tensor((2073600, 3), dtype="float16"))
            R.output(lv5)
        return lv5

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
mod = Module
extractor = TileGraphExtractor(mod)
extractor.visit_expr(mod["main"])

ordered_nodes = extractor.ordered_nodes
node_map = extractor.node_map #节点名称->relax call的映射


fusion_pipeline = FusionPipeline(ordered_nodes)
fusion_pipeline.build_fusion_group(ordered_nodes[0])