# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def fused_matmul_relu(x: T.Buffer((T.int64(128), T.int64(128)), "float32"), w: T.Buffer((T.int64(128), T.int64(128)), "float32"), compute_intermediate: T.Buffer((T.int64(128), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul_intermediate = T.alloc_buffer((T.int64(128), T.int64(128)))
        for i0, i1, k in T.grid(T.int64(128), T.int64(128), T.int64(128)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(x[v_i0, v_k], w[v_k, v_i1])
                T.writes(matmul_intermediate[v_i0, v_i1])
                with T.init():
                    matmul_intermediate[v_i0, v_i1] = T.float32(0)
                matmul_intermediate[v_i0, v_i1] = matmul_intermediate[v_i0, v_i1] + x[v_i0, v_k] * w[v_k, v_i1]
        for i0, i1 in T.grid(T.int64(128), T.int64(128)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(matmul_intermediate[v_i0, v_i1])
                T.writes(compute_intermediate[v_i0, v_i1])
                compute_intermediate[v_i0, v_i1] = T.max(matmul_intermediate[v_i0, v_i1], T.float32(0))

    @R.function
    def main(x: R.Tensor((128, 128), dtype="float32"), w: R.Tensor((128, 128), dtype="float32")) -> R.Tensor((128, 128), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.fused_matmul_relu, (x, w), out_sinfo=R.Tensor((128, 128), dtype="float32"))
            R.output(gv)
        return gv