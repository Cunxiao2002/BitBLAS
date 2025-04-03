# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def fused_fused_dense_relu_0_fused_dense_relu_1(input0: T.Buffer((T.int64(2073600), T.int64(64)), "float16"), param_0: T.Buffer((T.int64(64), T.int64(64)), "float16"), param_1: T.Buffer((T.int64(64), T.int64(64)), "float16"), T_relu_intermediate_intermediate_1: T.Buffer((T.int64(2073600), T.int64(64)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        input0_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(2073600), T.int64(64)), "float16", scope="shared.dyn")
        param_0_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(64)), "float16", scope="shared.dyn")
        input0_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), T.int64(129600), T.int64(4), T.int64(32), T.int64(8)), "float16", scope="warp")
        param_0_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), T.int64(4), T.int64(4), T.int64(32), T.int64(8)), "float16", scope="warp")
        T_matmul_NT_intermediate_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(2073600), T.int64(64)), "float16", scope="shared.dyn")
        T_matmul_NT_intermediate_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), T.int64(129600), T.int64(4), T.int64(32), T.int64(8)), "float16", scope="warp")
        T_relu_intermediate_intermediate = T.alloc_buffer((T.int64(2073600), T.int64(64)), "float16")
        input0_reindex_shared_dyn_1 = T.alloc_buffer((T.int64(1), T.int64(2073600), T.int64(64)), "float16", scope="shared.dyn")
        param_0_reindex_shared_dyn_1 = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(64)), "float16", scope="shared.dyn")
        input0_reindex_shared_dyn_warp_1 = T.alloc_buffer((T.int64(1), T.int64(129600), T.int64(4), T.int64(32), T.int64(8)), "float16", scope="warp")
        param_0_reindex_shared_dyn_warp_1 = T.alloc_buffer((T.int64(1), T.int64(4), T.int64(4), T.int64(32), T.int64(8)), "float16", scope="warp")
        T_matmul_NT_intermediate_reindex_shared_dyn_1 = T.alloc_buffer((T.int64(1), T.int64(2073600), T.int64(64)), "float16", scope="shared.dyn")
        T_matmul_NT_intermediate_reindex_shared_dyn_warp_1 = T.alloc_buffer((T.int64(1), T.int64(129600), T.int64(4), T.int64(32), T.int64(8)), "float16", scope="warp")
        for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding(T.int64(21600), thread="blockIdx.y"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                    for ax1_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax2_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                            for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(3), T.int64(2)):
                                with T.block("T_matmul_NT_o_init"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0)
                                    v1_o = T.axis.spatial(T.int64(129600), ax1_0_0_ax2_0_0_fused * T.int64(6) + ax1_0_2 * T.int64(3) + ax1_0_3_init)
                                    v2_o = T.axis.spatial(T.int64(4), ax2_0_2 * T.int64(2) + ax2_0_3_init)
                                    T.reads()
                                    T.writes(T_matmul_NT_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    with T.block("T_matmul_NT_init_o"):
                                        v1_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        T.reads()
                                        T.writes(T_matmul_NT_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        C_warp = T.match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=1)
                                        for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                            T.mma_fill("float16", 8, C_warp.data, C_warp.elem_offset)
                            for ax3_0_0 in range(T.int64(2)):
                                for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                                        for ax0_ax1_ax2_fused_2 in T.unroll(T.int64(3), annotations={"pragma_unroll_explicit": 0}):
                                            for ax0_ax1_ax2_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                                for ax0_ax1_ax2_fused_4 in T.vectorized(T.int64(8)):
                                                    with T.block("input0_reindex_shared.dyn"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(2073600), ax1_0_0_ax2_0_0_fused * T.int64(96) + (ax0_ax1_ax2_fused_0 * T.int64(1536) + ax0_ax1_ax2_fused_1 * T.int64(768) + ax0_ax1_ax2_fused_2 * T.int64(256) + ax0_ax1_ax2_fused_3 * T.int64(8) + ax0_ax1_ax2_fused_4) // T.int64(32))
                                                        v2 = T.axis.spatial(T.int64(64), ax3_0_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(1536) + ax0_ax1_ax2_fused_1 * T.int64(768) + ax0_ax1_ax2_fused_2 * T.int64(256) + ax0_ax1_ax2_fused_3 * T.int64(8) + ax0_ax1_ax2_fused_4) % T.int64(32))
                                                        T.reads(input0[v1, v2])
                                                        T.writes(input0_reindex_shared_dyn[v0, v1, v2])
                                                        T.block_attr({"permuted_layout": 1})
                                                        input0_reindex_shared_dyn[v0, v1, v2] = input0[v1, v2]
                                for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                                        for ax0_ax1_ax2_fused_2 in T.unroll(T.int64(2), annotations={"pragma_unroll_explicit": 0}):
                                            for ax0_ax1_ax2_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                                for ax0_ax1_ax2_fused_4 in T.vectorized(T.int64(8)):
                                                    with T.block("param_0_reindex_shared.dyn"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1 * T.int64(512) + ax0_ax1_ax2_fused_2 * T.int64(256) + ax0_ax1_ax2_fused_3 * T.int64(8) + ax0_ax1_ax2_fused_4) // T.int64(32))
                                                        v2 = T.axis.spatial(T.int64(64), ax3_0_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1 * T.int64(512) + ax0_ax1_ax2_fused_2 * T.int64(256) + ax0_ax1_ax2_fused_3 * T.int64(8) + ax0_ax1_ax2_fused_4) % T.int64(32))
                                                        T.reads(param_0[v1, v2])
                                                        T.writes(param_0_reindex_shared_dyn[v0, v1, v2])
                                                        T.block_attr({"permuted_layout": 1})
                                                        param_0_reindex_shared_dyn[v0, v1, v2] = param_0[v1, v2]
                                for ax3_0_1 in range(T.int64(2)):
                                    for ax0_0, ax1_0 in T.grid(T.int64(3), T.int64(1)):
                                        with T.block("input0_reindex_shared.dyn_warp_o"):
                                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1_o = T.axis.spatial(T.int64(129600), ax1_0_0_ax2_0_0_fused * T.int64(6) + ax1_0_2 * T.int64(3) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(4), ax3_0_0 * T.int64(2) + ax3_0_1 + ax1_0)
                                            T.reads(input0_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(input0_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                            T.block_attr({"permuted_layout": 1})
                                            warp = T.match_buffer(input0_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                            shared = T.match_buffer(input0_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                            for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                                T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + T.int64(8) * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * T.int64(16), 1), shared.strides[0] * (tx % T.int64(16)) + T.int64(8) * (tx // T.int64(16)))
                                    for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(1)):
                                        with T.block("param_0_reindex_shared.dyn_warp_o"):
                                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1_o = T.axis.spatial(T.int64(4), ax2_0_2 * T.int64(2) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(4), ax3_0_0 * T.int64(2) + ax3_0_1 + ax1_0)
                                            T.reads(param_0_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(param_0_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                            T.block_attr({"permuted_layout": 1})
                                            warp = T.match_buffer(param_0_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                            shared = T.match_buffer(param_0_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                            for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                                T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + T.int64(8) * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * T.int64(16), 1), shared.strides[0] * T.int64(8) * (tx // T.int64(16)) + shared.strides[0] * (tx % T.int64(8)) + T.int64(8) * (tx % T.int64(16) // T.int64(8)))
                                    for ax1_0_3, ax2_0_3 in T.grid(T.int64(3), T.int64(2)):
                                        with T.block("T_matmul_NT_o_update"):
                                            v0_o = T.axis.spatial(T.int64(1), ax0)
                                            v1_o = T.axis.spatial(T.int64(129600), ax1_0_0_ax2_0_0_fused * T.int64(6) + ax1_0_2 * T.int64(3) + ax1_0_3)
                                            v2_o = T.axis.spatial(T.int64(4), ax2_0_2 * T.int64(2) + ax2_0_3)
                                            v3_o = T.axis.reduce(T.int64(4), ax3_0_0 * T.int64(2) + ax3_0_1)
                                            T.reads(T_matmul_NT_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], input0_reindex_shared_dyn_warp[T.int64(0), v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], param_0_reindex_shared_dyn_warp[T.int64(0), v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                            T.writes(T_matmul_NT_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                            with T.block("T_matmul_NT_o"):
                                                v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                                v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                                v3_i_o = T.axis.reduce(T.int64(1), T.int64(0))
                                                T.reads(T_matmul_NT_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], input0_reindex_shared_dyn_warp[T.int64(0), v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], param_0_reindex_shared_dyn_warp[T.int64(0), v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                                T.writes(T_matmul_NT_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                                A = T.match_buffer(input0_reindex_shared_dyn_warp[T.int64(0), v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                                B = T.match_buffer(param_0_reindex_shared_dyn_warp[T.int64(0), v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                                C = T.match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                                for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A.data, A.elem_offset + tx * T.int64(8), B.data, B.elem_offset + tx * T.int64(8), C.data, C.elem_offset + tx * T.int64(8), T.bool(False))
                                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A.data, A.elem_offset + tx * T.int64(8), B.data, B.elem_offset + tx * T.int64(8) + T.int64(4), C.data, C.elem_offset + tx * T.int64(8) + T.int64(4), T.bool(False))
                            for ax0_0, ax1_0 in T.grid(T.int64(3), T.int64(2)):
                                with T.block("T_matmul_NT_intermediate_reindex_shared.dyn_warp_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(129600), ax1_0_0_ax2_0_0_fused * T.int64(6) + ax1_0_2 * T.int64(3) + ax0_0)
                                    v2_o = T.axis.spatial(T.int64(4), ax2_0_2 * T.int64(2) + ax1_0)
                                    T.reads(T_matmul_NT_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.writes(T_matmul_NT_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    C_warp = T.match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=1)
                                    C = T.match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=1)
                                    for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        T.mma_store("float16", 16, 16, T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(16), 2), C_warp.data, C_warp.elem_offset, C.strides[0])
                        for ax0_ax1_ax2_fused_0 in T.unroll(T.int64(12), annotations={"pragma_unroll_explicit": 0}):
                            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_ax2_fused_2 in T.vectorized(T.int64(8)):
                                    with T.block("T_matmul_NT_intermediate_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(2073600), ax1_0_0_ax2_0_0_fused * T.int64(96) + ax1_0_2 * T.int64(48) + (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2) // T.int64(64))
                                        v2 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2) % T.int64(64))
                                        T.reads(T_matmul_NT_intermediate_reindex_shared_dyn[v0, v1, v2])
                                        T.writes(T_relu_intermediate_intermediate[v1, v2])
                                        T_relu_intermediate_intermediate[v1, v2] = T.max(T_matmul_NT_intermediate_reindex_shared_dyn[v0, v1, v2], T.float16(0))
                        ax1_0_0_ax2_0_0_fused_1 = T.int64()
                        ax1_0_2_1 = T.int64()
                        for ax2_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                            ax0_1 = T.int64()
                            for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(3), T.int64(2)):
                                with T.block("T_matmul_NT_o_init_1"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0_1)
                                    v1_o = T.axis.spatial(T.int64(129600), ax1_0_0_ax2_0_0_fused_1 * T.int64(6) + ax1_0_2_1 * T.int64(3) + ax1_0_3_init)
                                    v2_o = T.axis.spatial(T.int64(4), ax2_0_2 * T.int64(2) + ax2_0_3_init)
                                    T.reads()
                                    T.writes(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    with T.block("T_matmul_NT_init_o_1"):
                                        v1_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        T.reads()
                                        T.writes(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        C_warp = T.match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=1)
                                        for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                            T.mma_fill("float16", 8, C_warp.data, C_warp.elem_offset)
                            for ax3_0_0 in range(T.int64(2)):
                                for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                                        for ax0_ax1_ax2_fused_2 in T.unroll(T.int64(2), annotations={"pragma_unroll_explicit": 0}):
                                            for ax0_ax1_ax2_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                                for ax0_ax1_ax2_fused_4 in T.vectorized(T.int64(8)):
                                                    with T.block("param_0_reindex_shared.dyn_1"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1 * T.int64(512) + ax0_ax1_ax2_fused_2 * T.int64(256) + ax0_ax1_ax2_fused_3 * T.int64(8) + ax0_ax1_ax2_fused_4) // T.int64(32))
                                                        v2 = T.axis.spatial(T.int64(64), ax3_0_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1 * T.int64(512) + ax0_ax1_ax2_fused_2 * T.int64(256) + ax0_ax1_ax2_fused_3 * T.int64(8) + ax0_ax1_ax2_fused_4) % T.int64(32))
                                                        T.reads(param_1[v1, v2])
                                                        T.writes(param_0_reindex_shared_dyn_1[v0, v1, v2])
                                                        T.block_attr({"permuted_layout": 1})
                                                        param_0_reindex_shared_dyn_1[v0, v1, v2] = param_1[v1, v2]
                                for ax3_0_1 in range(T.int64(2)):
                                    for ax0_0, ax1_0 in T.grid(T.int64(3), T.int64(1)):
                                        with T.block("input0_reindex_shared.dyn_warp_o_1"):
                                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1_o = T.axis.spatial(T.int64(129600), ax1_0_0_ax2_0_0_fused_1 * T.int64(6) + ax1_0_2_1 * T.int64(3) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(4), ax3_0_0 * T.int64(2) + ax3_0_1 + ax1_0)
                                            T.reads(input0_reindex_shared_dyn_1[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(input0_reindex_shared_dyn_warp_1[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                            T.block_attr({"permuted_layout": 1})
                                            warp = T.match_buffer(input0_reindex_shared_dyn_warp_1[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                            shared = T.match_buffer(input0_reindex_shared_dyn_1[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                            for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                                T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + T.int64(8) * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * T.int64(16), 1), shared.strides[0] * (tx % T.int64(16)) + T.int64(8) * (tx // T.int64(16)))
                                    for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(1)):
                                        with T.block("param_0_reindex_shared.dyn_warp_o_1"):
                                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1_o = T.axis.spatial(T.int64(4), ax2_0_2 * T.int64(2) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(4), ax3_0_0 * T.int64(2) + ax3_0_1 + ax1_0)
                                            T.reads(param_0_reindex_shared_dyn_1[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(param_0_reindex_shared_dyn_warp_1[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                            T.block_attr({"permuted_layout": 1})
                                            warp = T.match_buffer(param_0_reindex_shared_dyn_warp_1[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                            shared = T.match_buffer(param_0_reindex_shared_dyn_1[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                            for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                                T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + T.int64(8) * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * T.int64(16), 1), shared.strides[0] * T.int64(8) * (tx // T.int64(16)) + shared.strides[0] * (tx % T.int64(8)) + T.int64(8) * (tx % T.int64(16) // T.int64(8)))
                                    for ax1_0_3, ax2_0_3 in T.grid(T.int64(3), T.int64(2)):
                                        with T.block("T_matmul_NT_o_update_1"):
                                            v0_o = T.axis.spatial(T.int64(1), ax0_1)
                                            v1_o = T.axis.spatial(T.int64(129600), ax1_0_0_ax2_0_0_fused_1 * T.int64(6) + ax1_0_2_1 * T.int64(3) + ax1_0_3)
                                            v2_o = T.axis.spatial(T.int64(4), ax2_0_2 * T.int64(2) + ax2_0_3)
                                            v3_o = T.axis.reduce(T.int64(4), ax3_0_0 * T.int64(2) + ax3_0_1)
                                            T.reads(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], input0_reindex_shared_dyn_warp_1[T.int64(0), v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], param_0_reindex_shared_dyn_warp_1[T.int64(0), v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                            T.writes(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                            with T.block("T_matmul_NT_o_1"):
                                                v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                                v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                                v3_i_o = T.axis.reduce(T.int64(1), T.int64(0))
                                                T.reads(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], input0_reindex_shared_dyn_warp_1[T.int64(0), v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], param_0_reindex_shared_dyn_warp_1[T.int64(0), v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                                T.writes(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                                A = T.match_buffer(input0_reindex_shared_dyn_warp_1[T.int64(0), v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                                B = T.match_buffer(param_0_reindex_shared_dyn_warp_1[T.int64(0), v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                                C = T.match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                                for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A.data, A.elem_offset + tx * T.int64(8), B.data, B.elem_offset + tx * T.int64(8), C.data, C.elem_offset + tx * T.int64(8), T.bool(False))
                                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A.data, A.elem_offset + tx * T.int64(8), B.data, B.elem_offset + tx * T.int64(8) + T.int64(4), C.data, C.elem_offset + tx * T.int64(8) + T.int64(4), T.bool(False))
                            for ax0_0, ax1_0 in T.grid(T.int64(3), T.int64(2)):
                                with T.block("T_matmul_NT_intermediate_reindex_shared.dyn_warp_o_1"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(129600), ax1_0_0_ax2_0_0_fused_1 * T.int64(6) + ax1_0_2_1 * T.int64(3) + ax0_0)
                                    v2_o = T.axis.spatial(T.int64(4), ax2_0_2 * T.int64(2) + ax1_0)
                                    T.reads(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.writes(T_matmul_NT_intermediate_reindex_shared_dyn_1[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    C_warp = T.match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=1)
                                    C = T.match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_1[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=1)
                                    for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        T.mma_store("float16", 16, 16, T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(16), 2), C_warp.data, C_warp.elem_offset, C.strides[0])
                        for ax0_ax1_ax2_fused_0 in T.unroll(T.int64(12), annotations={"pragma_unroll_explicit": 0}):
                            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_ax2_fused_2 in T.vectorized(T.int64(8)):
                                    with T.block("T_matmul_NT_intermediate_reindex_shared.dyn_1"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(2073600), ax1_0_0_ax2_0_0_fused_1 * T.int64(96) + ax1_0_2_1 * T.int64(48) + (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2) // T.int64(64))
                                        v2 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2) % T.int64(64))
                                        T.reads(T_matmul_NT_intermediate_reindex_shared_dyn_1[v0, v1, v2])
                                        T.writes(T_relu_intermediate_intermediate_1[v1, v2])
                                        T_relu_intermediate_intermediate_1[v1, v2] = T.max(T_matmul_NT_intermediate_reindex_shared_dyn_1[v0, v1, v2], T.float16(0))