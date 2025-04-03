# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def matmul(x: T.Buffer((T.int64(128), T.int64(128)), "float32"), w: T.Buffer((T.int64(128), T.int64(128)), "float32"), matmul: T.Buffer((T.int64(128), T.int64(128)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, k in T.grid(T.int64(128), T.int64(128), T.int64(128)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(x[v_i0, v_k], w[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + x[v_i0, v_k] * w[v_k, v_i1]

    @T.prim_func(private=True)
    def relu(lv0: T.Buffer((T.int64(128), T.int64(128)), "float32"), compute: T.Buffer((T.int64(128), T.int64(128)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(128), T.int64(128)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv0[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.max(lv0[v_i0, v_i1], T.float32(0))

    @R.function
    def main(x: R.Tensor((128, 128), dtype="float32"), w: R.Tensor((128, 128), dtype="float32")) -> R.Tensor((128, 128), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv0 = R.call_tir(cls.matmul, (x, w), out_sinfo=R.Tensor((128, 128), dtype="float32"))
            lv1 = R.call_tir(cls.relu, (lv0,), out_sinfo=R.Tensor((128, 128), dtype="float32"))
            R.output(lv1)
        return lv1