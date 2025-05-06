@I.ir_module
class Module:
    @T.prim_func
    def fused_dense_relu_0(input0: T.Buffer((T.int64(2073600), T.int64(64)), "float16"), param_0: T.Buffer((T.int64(64), T.int64(64)), "float16"), T_relu_intermediate: T.Buffer((T.int64(2073600), T.int64(64)), "float16")):
        T.func_attr({"dlight.tensorcore_prenormlized": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_matmul_NT_intermediate_reindex_local = T.alloc_buffer((T.int64(1), T.int64(2073600), T.int64(64)), scope="local")
        for ax0_0_ax1_0_fused in T.thread_binding(T.int64(32400), thread="blockIdx.x"):
            for ax1_1_0_0 in T.thread_binding(T.int64(4), thread="vthread.x"):
                for ax0_1_0 in T.thread_binding(T.int64(4), thread="vthread.y"):
                    for ax0_1_1_ax1_1_0_1_fused in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                        for ax1_1_1_init in T.unroll(T.int64(2)):
                            with T.block("T_matmul_NT_init"):
                                v0 = T.axis.spatial(T.int64(2073600), ax0_0_ax1_0_fused * T.int64(64) + ax0_1_0 * T.int64(16) + ax0_1_1_ax1_1_0_1_fused // T.int64(8))
                                v1 = T.axis.spatial(T.int64(64), ax1_1_0_0 * T.int64(16) + ax0_1_1_ax1_1_0_1_fused % T.int64(8) * T.int64(2) + ax1_1_1_init)
                                T.reads()
                                T.writes(T_matmul_NT_intermediate_reindex_local[T.int64(0), v0, v1])
                                T_matmul_NT_intermediate_reindex_local[T.int64(0), v0, v1] = T.float32(0)
                        for ax2_0, ax2_1 in T.grid(T.int64(1), T.int64(64)):
                            for ax1_1_1 in T.unroll(T.int64(2)):
                                with T.block("T_matmul_NT_update"):
                                    v0 = T.axis.spatial(T.int64(2073600), ax0_0_ax1_0_fused * T.int64(64) + ax0_1_0 * T.int64(16) + ax0_1_1_ax1_1_0_1_fused // T.int64(8))
                                    v1 = T.axis.spatial(T.int64(64), ax1_1_0_0 * T.int64(16) + ax0_1_1_ax1_1_0_1_fused % T.int64(8) * T.int64(2) + ax1_1_1)
                                    v2 = T.axis.reduce(T.int64(64), ax2_0 * T.int64(64) + ax2_1)
                                    T.reads(T_matmul_NT_intermediate_reindex_local[T.int64(0), v0, v1], input0[v0, v2], param_0[v1, v2])
                                    T.writes(T_matmul_NT_intermediate_reindex_local[T.int64(0), v0, v1])
                                    T_matmul_NT_intermediate_reindex_local[T.int64(0), v0, v1] = T_matmul_NT_intermediate_reindex_local[T.int64(0), v0, v1] + T.Cast("float32", input0[v0, v2] * param_0[v1, v2])
                        for ax0 in T.unroll(T.int64(2)):
                            with T.block("T_matmul_NT_intermediate_reindex_local"):
                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v1 = T.axis.spatial(T.int64(2073600), ax0_0_ax1_0_fused * T.int64(64) + ax0_1_0 * T.int64(16) + ax0_1_1_ax1_1_0_1_fused // T.int64(8))
                                v2 = T.axis.spatial(T.int64(64), ax1_1_0_0 * T.int64(16) + ax0_1_1_ax1_1_0_1_fused % T.int64(8) * T.int64(2) + ax0)
                                T.reads(T_matmul_NT_intermediate_reindex_local[v0, v1, v2])
                                T.writes(T_relu_intermediate[v1, v2])
                                T_relu_intermediate[v1, v2] = T.Cast("float16", T.max(T_matmul_NT_intermediate_reindex_local[v0, v1, v2], T.float32(0)))
    
    @T.prim_func
    def fused_dense_relu_1(input0: T.Buffer((T.int64(2073600), T.int64(64)), "float16"), param_0: T.Buffer((T.int64(64), T.int64(64)), "float16"), T_relu_intermediate: T.Buffer((T.int64(2073600), T.int64(64)), "float16")):
        T.func_attr({"dlight.tensorcore_prenormlized": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_matmul_NT_intermediate_reindex_local = T.alloc_buffer((T.int64(1), T.int64(2073600), T.int64(64)), scope="local")
        for ax0_0_ax1_0_fused in T.thread_binding(T.int64(32400), thread="blockIdx.x"):
            for ax1_1_0_0 in T.thread_binding(T.int64(4), thread="vthread.x"):
                for ax0_1_0 in T.thread_binding(T.int64(4), thread="vthread.y"):
                    for ax0_1_1_ax1_1_0_1_fused in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                        for ax1_1_1_init in T.unroll(T.int64(2)):
                            with T.block("T_matmul_NT_init"):
                                v0 = T.axis.spatial(T.int64(2073600), ax0_0_ax1_0_fused * T.int64(64) + ax0_1_0 * T.int64(16) + ax0_1_1_ax1_1_0_1_fused // T.int64(8))
                                v1 = T.axis.spatial(T.int64(64), ax1_1_0_0 * T.int64(16) + ax0_1_1_ax1_1_0_1_fused % T.int64(8) * T.int64(2) + ax1_1_1_init)
                                T.reads()
                                T.writes(T_matmul_NT_intermediate_reindex_local[T.int64(0), v0, v1])
                                T_matmul_NT_intermediate_reindex_local[T.int64(0), v0, v1] = T.float32(0)
                        for ax2_0, ax2_1 in T.grid(T.int64(1), T.int64(64)):
                            for ax1_1_1 in T.unroll(T.int64(2)):
                                with T.block("T_matmul_NT_update"):
                                    v0 = T.axis.spatial(T.int64(2073600), ax0_0_ax1_0_fused * T.int64(64) + ax0_1_0 * T.int64(16) + ax0_1_1_ax1_1_0_1_fused // T.int64(8))
                                    v1 = T.axis.spatial(T.int64(64), ax1_1_0_0 * T.int64(16) + ax0_1_1_ax1_1_0_1_fused % T.int64(8) * T.int64(2) + ax1_1_1)
                                    v2 = T.axis.reduce(T.int64(64), ax2_0 * T.int64(64) + ax2_1)
                                    T.reads(T_matmul_NT_intermediate_reindex_local[T.int64(0), v0, v1], input0[v0, v2], param_0[v1, v2])
                                    T.writes(T_matmul_NT_intermediate_reindex_local[T.int64(0), v0, v1])
                                    T_matmul_NT_intermediate_reindex_local[T.int64(0), v0, v1] = T_matmul_NT_intermediate_reindex_local[T.int64(0), v0, v1] + T.Cast("float32", input0[v0, v2] * param_0[v1, v2])
                        for ax0 in T.unroll(T.int64(2)):
                            with T.block("T_matmul_NT_intermediate_reindex_local"):
                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v1 = T.axis.spatial(T.int64(2073600), ax0_0_ax1_0_fused * T.int64(64) + ax0_1_0 * T.int64(16) + ax0_1_1_ax1_1_0_1_fused // T.int64(8))
                                v2 = T.axis.spatial(T.int64(64), ax1_1_0_0 * T.int64(16) + ax0_1_1_ax1_1_0_1_fused % T.int64(8) * T.int64(2) + ax0)
                                T.reads(T_matmul_NT_intermediate_reindex_local[v0, v1, v2])
                                T.writes(T_relu_intermediate[v1, v2])
                                T_relu_intermediate[v1, v2] = T.Cast("float16", T.max(T_matmul_NT_intermediate_reindex_local[v0, v1, v2], T.float32(0)))