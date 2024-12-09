import tvm
from tvm.script import tir as T
import bitblas

bitblas.set_log_level("DEBUG")

@tvm.script.ir_module
class FusedSingleOp:
    @T.prim_func(private=True)
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

ir_module = FusedSingleOp
target = tvm.target.Target("cuda")
    
with target:
    mod = bitblas.ApplyFastTuning(topk=1)(ir_module)

print(mod)
from tvm import relax
exec = relax.build(mod, target="cuda")
dev = tvm.device("cuda", 0)
vm = relax.VirtualMachine(exec, dev)
