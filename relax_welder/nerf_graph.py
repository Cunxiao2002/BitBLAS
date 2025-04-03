from base.arch import auto_infer_current_arch
import tvm
from tvm import target
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
import bitblas
from bitblas.base.arch import auto_infer_current_arch
from tvm import relax

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def dense1(lv11: T.Buffer((T.int64(2073600), T.int64(64)), "float32"), B: T.Buffer((T.int64(3), T.int64(64)), "float32"), T_matmul_NT: T.Buffer((T.int64(2073600), T.int64(3)), "float32")):
        T.func_attr({"layout_free_buffers": [1], "op_attrs": {"op_name": "nn.dense", "out_dtype": "float32", "units": None}, "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, k in T.grid(T.int64(2073600), T.int64(3), T.int64(64)):
            with T.block("T_matmul_NT"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(lv11[v_i0, v_k], B[v_i1, v_k])
                T.writes(T_matmul_NT[v_i0, v_i1])
                with T.init():
                    T_matmul_NT[v_i0, v_i1] = T.float32(0)
                T_matmul_NT[v_i0, v_i1] = T_matmul_NT[v_i0, v_i1] + lv11[v_i0, v_k] * B[v_i1, v_k]

    @T.prim_func(private=True)
    def fused_dense_relu(input0: T.Buffer((T.int64(2073600), T.int64(64)), "float32"), param_0: T.Buffer((T.int64(64), T.int64(64)), "float32"), T_relu_intermediate: T.Buffer((T.int64(2073600), T.int64(64)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_matmul_NT_intermediate = T.alloc_buffer((T.int64(2073600), T.int64(64)))
        for i0, i1, k in T.grid(T.int64(2073600), T.int64(64), T.int64(64)):
            with T.block("T_matmul_NT"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(input0[v_i0, v_k], param_0[v_i1, v_k])
                T.writes(T_matmul_NT_intermediate[v_i0, v_i1])
                with T.init():
                    T_matmul_NT_intermediate[v_i0, v_i1] = T.float32(0)
                T_matmul_NT_intermediate[v_i0, v_i1] = T_matmul_NT_intermediate[v_i0, v_i1] + input0[v_i0, v_k] * param_0[v_i1, v_k]
        for ax0, ax1 in T.grid(T.int64(2073600), T.int64(64)):
            with T.block("T_relu"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(T_matmul_NT_intermediate[v_ax0, v_ax1])
                T.writes(T_relu_intermediate[v_ax0, v_ax1])
                T_relu_intermediate[v_ax0, v_ax1] = T.max(T_matmul_NT_intermediate[v_ax0, v_ax1], T.float32(0))

    @R.function
    def main(input0: R.Tensor((2073600, 64), dtype="float32")) -> R.Tensor((2073600, 3), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.fused_dense_relu, (input0, metadata["relax.expr.Constant"][0]), out_sinfo=R.Tensor((2073600, 64), dtype="float32"))
            lv1 = R.call_tir(cls.fused_dense_relu, (lv, metadata["relax.expr.Constant"][1]), out_sinfo=R.Tensor((2073600, 64), dtype="float32"))
            lv2 = R.call_tir(cls.fused_dense_relu, (lv1, metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((2073600, 64), dtype="float32"))
            lv3 = R.call_tir(cls.fused_dense_relu, (lv2, metadata["relax.expr.Constant"][3]), out_sinfo=R.Tensor((2073600, 64), dtype="float32"))
            lv4 = R.call_tir(cls.fused_dense_relu, (lv3, metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((2073600, 64), dtype="float32"))
            lv5 = R.call_tir(cls.fused_dense_relu, (lv4, metadata["relax.expr.Constant"][5]), out_sinfo=R.Tensor((2073600, 64), dtype="float32"))
            gv = R.call_tir(cls.dense1, (lv5, metadata["relax.expr.Constant"][6]), out_sinfo=R.Tensor((2073600, 3), dtype="float32"))
            R.output(gv)
        return gv


@ir.transform.module_pass(opt_level=0)
class CustomAnnotateTIROpPattern:
    def transform_module(self, mod, ctx):
        @tvm.tir.transform.prim_func_pass(opt_level=0)
        def transform(func, mod, ctx):
            return func.with_attr("op_pattern", 0)  # 将所有 TIR 标记为 kElemWise
        return transform(mod)


arch = auto_infer_current_arch()
# func = mod["fused_dense_relu"]
# tensorized_func, tags0 = bitblas.gpu.matmul_analysis.get_tensorized_func_and_tags(func, arch.target)
# print(f"tensorized_func is \n{tensorized_func}")

class TileGraphExtractor(relax.PyExprVisitor):
    def __init__(self):
        super().__init__()
        self.target = target
        self.ordered_nodes = []
        self.node_map = {}
    
    class NameExtractor(relax.PyExprVisitor):
        def __init__(self):
            super().__init__()
            self.op_names = []

        def visit_call_(self, call):
            super().visit_call_(call)
            name = call.op.name.replace(".", "_")
            self.op_names.append(name)

    
    def visit_call_(self, call):
        pass

