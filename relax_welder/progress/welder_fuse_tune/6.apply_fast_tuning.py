# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def fused_matmul_relu(x: T.Buffer((T.int64(128), T.int64(128)), "float32"), w: T.Buffer((T.int64(128), T.int64(128)), "float32"), compute_intermediate: T.Buffer((T.int64(128), T.int64(128)), "float32")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul_intermediate_local = T.alloc_buffer((T.int64(128), T.int64(128)), scope="local")
        for ax0_0_ax1_0_fused in T.thread_binding(T.int64(128), thread="blockIdx.x"):
            for ax0_1_ax1_1_fused in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                with T.block("matmul_init"):
                    v0 = T.axis.spatial(T.int64(128), ax0_0_ax1_0_fused // T.int64(2) * T.int64(2) + ax0_1_ax1_1_fused // T.int64(64))
                    v1 = T.axis.spatial(T.int64(128), ax0_0_ax1_0_fused % T.int64(2) * T.int64(64) + ax0_1_ax1_1_fused % T.int64(64))
                    T.reads()
                    T.writes(matmul_intermediate_local[v0, v1])
                    matmul_intermediate_local[v0, v1] = T.float32(0)
                for ax2_0, ax2_1 in T.grid(T.int64(2), T.int64(64)):
                    with T.block("matmul_update"):
                        v0 = T.axis.spatial(T.int64(128), ax0_0_ax1_0_fused // T.int64(2) * T.int64(2) + ax0_1_ax1_1_fused // T.int64(64))
                        v1 = T.axis.spatial(T.int64(128), ax0_0_ax1_0_fused % T.int64(2) * T.int64(64) + ax0_1_ax1_1_fused % T.int64(64))
                        v2 = T.axis.reduce(T.int64(128), ax2_0 * T.int64(64) + ax2_1)
                        T.reads(matmul_intermediate_local[v0, v1], x[v0, v2], w[v2, v1])
                        T.writes(matmul_intermediate_local[v0, v1])
                        matmul_intermediate_local[v0, v1] = matmul_intermediate_local[v0, v1] + x[v0, v2] * w[v2, v1]
                with T.block("matmul_intermediate_local"):
                    v0 = T.axis.spatial(T.int64(128), ax0_0_ax1_0_fused // T.int64(2) * T.int64(2) + ax0_1_ax1_1_fused // T.int64(64))
                    v1 = T.axis.spatial(T.int64(128), ax0_0_ax1_0_fused % T.int64(2) * T.int64(64) + ax0_1_ax1_1_fused % T.int64(64))
                    T.reads(matmul_intermediate_local[v0, v1])
                    T.writes(compute_intermediate[v0, v1])
                    compute_intermediate[v0, v1] = T.max(matmul_intermediate_local[v0, v1], T.float32(0))

    @R.function
    def main(x: R.Tensor((128, 128), dtype="float32"), w: R.Tensor((128, 128), dtype="float32")) -> R.Tensor((128, 128), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.fused_matmul_relu, (x, w), out_sinfo=R.Tensor((128, 128), dtype="float32"))
            R.output(gv)
        return gv