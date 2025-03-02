from tvm.script import ir as I
from tvm.script import tir as T
import tvm
from tvm import tir
from tvm import ir, transform
from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def fused_gemm_0_gemm_1(A: T.Buffer((T.int64(512), T.int64(128)), "float16"), B: T.Buffer((T.int64(128), T.int64(512)), "float16"), D: T.Buffer((T.int64(512), T.int64(128)), "float16"), C_intermediate_1: T.Buffer((T.int64(512), T.int64(128)), "float16")):
        T.func_attr({"private": T.bool(False), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_reindex_shared_dyn = T.alloc_buffer((1, 512, 128), "float16", scope="shared.dyn")
        B_reindex_shared_dyn = T.alloc_buffer((1, 128, 512), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_warp = T.alloc_buffer((1, 32, 8, 32, 8), "float16", scope="warp")
        B_reindex_shared_dyn_warp = T.alloc_buffer((1, 8, 32, 32, 8), "float16", scope="warp")
        C_reindex_shared_dyn = T.alloc_buffer((1, 512, 512), "float16", scope="shared.dyn")
        C_reindex_shared_dyn_warp = T.alloc_buffer((1, 32, 32, 32, 8), "float16", scope="warp")
        C_intermediate = T.alloc_buffer((T.int64(512), T.int64(512)), "float16")
        A_reindex_shared_dyn_1 = T.alloc_buffer((1, 512, 512), "float16", scope="shared.dyn")
        B_reindex_shared_dyn_1 = T.alloc_buffer((1, 512, 128), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_warp_1 = T.alloc_buffer((1, 32, 32, 32, 8), "float16", scope="warp")
        B_reindex_shared_dyn_warp_1 = T.alloc_buffer((1, 32, 8, 32, 8), "float16", scope="warp")
        C_reindex_shared_dyn_1 = T.alloc_buffer((1, 512, 128), "float16", scope="shared.dyn")
        C_reindex_shared_dyn_warp_1 = T.alloc_buffer((1, 32, 8, 32, 8), "float16", scope="warp")
        for ax0 in T.thread_binding(1, thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding(32, thread="blockIdx.y"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(1, thread="blockIdx.x"):
                    for ax1_0_2 in T.thread_binding(1, thread="threadIdx.y"):
                        for ax2_0_2 in T.thread_binding(4, thread="threadIdx.z"):
                            for ax1_0_3_init, ax2_0_3_init in T.grid(1, 8):
                                with T.block("gemm_o_init"):
                                    v0_o = T.axis.spatial(1, ax0)
                                    v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused + ax1_0_2 + ax1_0_3_init)
                                    v2_o = T.axis.spatial(32, ax2_0_2 * 8 + ax2_0_3_init)
                                    T.reads()
                                    T.writes(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8])
                                    with T.block("gemm_init_o"):
                                        v1_i_init_o = T.axis.spatial(1, 0)
                                        v2_i_init_o = T.axis.spatial(1, 0)
                                        T.reads()
                                        T.writes(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8])
                                        C_warp = T.match_buffer(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=1)
                                        for tx in T.thread_binding(32, thread="threadIdx.x"):
                                            T.mma_fill("float16", 8, C_warp.data, C_warp.elem_offset)
                            for ax3_0_0 in range(1):
                                for ax0_ax1_ax2_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                        for ax0_ax1_ax2_fused_2 in T.unroll(2, annotations={"pragma_unroll_explicit": 0}):
                                            for ax0_ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax0_ax1_ax2_fused_4 in T.vectorized(8):
                                                    with T.block("A_reindex_shared.dyn"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial(512, ax1_0_0_ax2_0_0_fused * 16 + (ax0_ax1_ax2_fused_0 * 2048 + ax0_ax1_ax2_fused_1 * 512 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) // 128)
                                                        v2 = T.axis.spatial(128, (ax0_ax1_ax2_fused_0 * 2048 + ax0_ax1_ax2_fused_1 * 512 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) % 128)
                                                        T.reads(A[v1, v2])
                                                        T.writes(A_reindex_shared_dyn[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "permuted_layout": 0})
                                                        A_reindex_shared_dyn[v0, v1, v2] = A[v1, v2]
                                for ax0_ax1_ax2_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                        for ax0_ax1_ax2_fused_2 in T.unroll(64, annotations={"pragma_unroll_explicit": 0}):
                                            for ax0_ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax0_ax1_ax2_fused_4 in T.vectorized(8):
                                                    with T.block("B_reindex_shared.dyn"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial(128, (ax0_ax1_ax2_fused_0 * 65536 + ax0_ax1_ax2_fused_1 * 16384 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) // 512)
                                                        v2 = T.axis.spatial(512, (ax0_ax1_ax2_fused_0 * 65536 + ax0_ax1_ax2_fused_1 * 16384 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) % 512)
                                                        T.reads(B[v1, v2])
                                                        T.writes(B_reindex_shared_dyn[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "permuted_layout": 0})
                                                        B_reindex_shared_dyn[v0, v1, v2] = B[v1, v2]
                                for ax3_0_1 in range(8):
                                    for ax0_0, ax1_0 in T.grid(1, 1):
                                        with T.block("A_reindex_shared.dyn_warp_o"):
                                            v0_o = T.axis.spatial(1, 0)
                                            v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused + ax0_0)
                                            v2_o = T.axis.spatial(8, ax3_0_1 + ax1_0)
                                            T.reads(A_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(A_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, 0:32, 0:8])
                                            T.block_attr({"permuted_layout": 0})
                                            warp = T.match_buffer(A_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                            shared = T.match_buffer(A_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                            for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + 8 * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * 16, 1), shared.strides[0] * (tx % 16) + 8 * (tx // 16))
                                    for ax0_0, ax1_0 in T.grid(1, 8):
                                        with T.block("B_reindex_shared.dyn_warp_o"):
                                            v0_o = T.axis.spatial(1, 0)
                                            v1_o = T.axis.spatial(8, ax3_0_1 + ax0_0)
                                            v2_o = T.axis.spatial(32, ax2_0_2 * 8 + ax1_0)
                                            T.reads(B_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(B_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, 0:32, 0:8])
                                            T.block_attr({"permuted_layout": 0})
                                            warp = T.match_buffer(B_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                            shared = T.match_buffer(B_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                            for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                T.ptx_ldmatrix("float16", T.bool(True), 4, ".b16", warp.data, warp.elem_offset + 8 * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * 16, 1), shared.strides[0] * (tx % 16) + 8 * (tx // 16))
                                    for ax1_0_3, ax2_0_3 in T.grid(1, 8):
                                        with T.block("gemm_o_update"):
                                            v0_o = T.axis.spatial(1, ax0)
                                            v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused + ax1_0_2 + ax1_0_3)
                                            v2_o = T.axis.spatial(32, ax2_0_2 * 8 + ax2_0_3)
                                            v3_o = T.axis.reduce(8, ax3_0_0 * 8 + ax3_0_1)
                                            T.reads(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8], A_reindex_shared_dyn_warp[0, v1_o, v3_o, 0:32, 0:8], B_reindex_shared_dyn_warp[0, v3_o, v2_o, 0:32, 0:8])
                                            T.writes(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8])
                                            with T.block("gemm_o"):
                                                v1_i_o = T.axis.spatial(1, 0)
                                                v2_i_o = T.axis.spatial(1, 0)
                                                v3_i_o = T.axis.reduce(1, 0)
                                                T.reads(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8], A_reindex_shared_dyn_warp[0, v1_o, v3_o, 0:32, 0:8], B_reindex_shared_dyn_warp[0, v3_o, v2_o, 0:32, 0:8])
                                                T.writes(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8])
                                                A_1 = T.match_buffer(A_reindex_shared_dyn_warp[0, v1_o, v3_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                B_1 = T.match_buffer(B_reindex_shared_dyn_warp[0, v3_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                C_1 = T.match_buffer(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data, A_1.elem_offset + tx * 8, B_1.data, B_1.elem_offset + tx * 8, C_1.data, C_1.elem_offset + tx * 8, T.bool(False))
                                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data, A_1.elem_offset + tx * 8, B_1.data, B_1.elem_offset + tx * 8 + 4, C_1.data, C_1.elem_offset + tx * 8 + 4, T.bool(False))
                            for ax0_0, ax1_0 in T.grid(1, 8):
                                with T.block("C_reindex_shared.dyn_warp_o"):
                                    v0_o = T.axis.spatial(1, 0)
                                    v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused)
                                    v2_o = T.axis.spatial(32, ax2_0_2 * 8 + ax1_0)
                                    T.reads(C_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, 0:32, 0:8])
                                    T.writes(C_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                    C_warp = T.match_buffer(C_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=1)
                                    C_1 = T.match_buffer(C_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_1_s0", "C_1_s1"), scope="shared.dyn", offset_factor=1)
                                    for tx in T.thread_binding(32, thread="threadIdx.x"):
                                        T.mma_store("float16", 16, 16, T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_1.strides[0] * 16, 2), C_warp.data, C_warp.elem_offset, C_1.strides[0])
                            ax1_0_0_ax2_0_0_fused_1 = T.int32()
                            ax1_0_2_1 = T.int32()
                            ax2_0_2_1 = T.int32()
                            for ax1_0_3_init, ax2_0_3_init in T.grid(1, 2):
                                with T.block("gemm_o_init_1"):
                                    v0_o = T.axis.spatial(1, ax0)
                                    v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused_1 + ax1_0_2_1 + ax1_0_3_init)
                                    v2_o = T.axis.spatial(8, ax2_0_2_1 * 2 + ax2_0_3_init)
                                    T.reads()
                                    T.writes(C_reindex_shared_dyn_warp_1[0, v1_o, v2_o, 0:32, 0:8])
                                    with T.block("gemm_init_o_1"):
                                        v1_i_init_o = T.axis.spatial(1, 0)
                                        v2_i_init_o = T.axis.spatial(1, 0)
                                        T.reads()
                                        T.writes(C_reindex_shared_dyn_warp_1[0, v1_o, v2_o, 0:32, 0:8])
                                        C_warp = T.match_buffer(C_reindex_shared_dyn_warp_1[0, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=1)
                                        for tx in T.thread_binding(32, thread="threadIdx.x"):
                                            T.mma_fill("float16", 8, C_warp.data, C_warp.elem_offset)
                            for ax3_0_0 in range(4):
                                for ax0_ax1_ax2_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                        for ax0_ax1_ax2_fused_2 in T.unroll(2, annotations={"pragma_unroll_explicit": 0}):
                                            for ax0_ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax0_ax1_ax2_fused_4 in T.vectorized(8):
                                                    with T.block("A_reindex_shared.dyn_1"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial(512, ax1_0_0_ax2_0_0_fused_1 * 16 + (ax0_ax1_ax2_fused_0 * 2048 + ax0_ax1_ax2_fused_1 * 512 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) // 128)
                                                        v2 = T.axis.spatial(512, ax3_0_0 * 128 + (ax0_ax1_ax2_fused_0 * 2048 + ax0_ax1_ax2_fused_1 * 512 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) % 128)
                                                        T.reads(C_intermediate[v1, v2])
                                                        T.writes(A_reindex_shared_dyn_1[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "permuted_layout": 0})
                                                        A_reindex_shared_dyn_1[v0, v1, v2] = C_intermediate[v1, v2]
                                for ax0_ax1_ax2_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                        for ax0_ax1_ax2_fused_2 in T.unroll(16, annotations={"pragma_unroll_explicit": 0}):
                                            for ax0_ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax0_ax1_ax2_fused_4 in T.vectorized(8):
                                                    with T.block("B_reindex_shared.dyn_1"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial(512, ax3_0_0 * 128 + (ax0_ax1_ax2_fused_0 * 16384 + ax0_ax1_ax2_fused_1 * 4096 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) // 128)
                                                        v2 = T.axis.spatial(128, (ax0_ax1_ax2_fused_0 * 16384 + ax0_ax1_ax2_fused_1 * 4096 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) % 128)
                                                        T.reads(D[v1, v2])
                                                        T.writes(B_reindex_shared_dyn_1[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "permuted_layout": 0})
                                                        B_reindex_shared_dyn_1[v0, v1, v2] = D[v1, v2]
                                for ax3_0_1 in range(8):
                                    for ax0_0, ax1_0 in T.grid(1, 1):
                                        with T.block("A_reindex_shared.dyn_warp_o_1"):
                                            v0_o = T.axis.spatial(1, 0)
                                            v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused_1 + ax0_0)
                                            v2_o = T.axis.spatial(32, ax3_0_0 * 8 + ax3_0_1 + ax1_0)
                                            T.reads(A_reindex_shared_dyn_1[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(A_reindex_shared_dyn_warp_1[v0_o, v1_o, v2_o, 0:32, 0:8])
                                            T.block_attr({"permuted_layout": 0})
                                            warp = T.match_buffer(A_reindex_shared_dyn_warp_1[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                            shared = T.match_buffer(A_reindex_shared_dyn_1[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                            for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + 8 * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * 16, 1), shared.strides[0] * (tx % 16) + 8 * (tx // 16))
                                    for ax0_0, ax1_0 in T.grid(1, 2):
                                        with T.block("B_reindex_shared.dyn_warp_o_1"):
                                            v0_o = T.axis.spatial(1, 0)
                                            v1_o = T.axis.spatial(32, ax3_0_0 * 8 + ax3_0_1 + ax0_0)
                                            v2_o = T.axis.spatial(8, ax2_0_2_1 * 2 + ax1_0)
                                            T.reads(B_reindex_shared_dyn_1[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(B_reindex_shared_dyn_warp_1[v0_o, v1_o, v2_o, 0:32, 0:8])
                                            T.block_attr({"permuted_layout": 0})
                                            warp = T.match_buffer(B_reindex_shared_dyn_warp_1[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                            shared = T.match_buffer(B_reindex_shared_dyn_1[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                            for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                T.ptx_ldmatrix("float16", T.bool(True), 4, ".b16", warp.data, warp.elem_offset + 8 * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * 16, 1), shared.strides[0] * (tx % 16) + 8 * (tx // 16))
                                    for ax1_0_3, ax2_0_3 in T.grid(1, 2):
                                        with T.block("gemm_o_update_1"):
                                            v0_o = T.axis.spatial(1, ax0)
                                            v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused_1 + ax1_0_2_1 + ax1_0_3)
                                            v2_o = T.axis.spatial(8, ax2_0_2_1 * 2 + ax2_0_3)
                                            v3_o = T.axis.reduce(32, ax3_0_0 * 8 + ax3_0_1)
                                            T.reads(C_reindex_shared_dyn_warp_1[0, v1_o, v2_o, 0:32, 0:8], A_reindex_shared_dyn_warp_1[0, v1_o, v3_o, 0:32, 0:8], B_reindex_shared_dyn_warp_1[0, v3_o, v2_o, 0:32, 0:8])
                                            T.writes(C_reindex_shared_dyn_warp_1[0, v1_o, v2_o, 0:32, 0:8])
                                            with T.block("gemm_o_1"):
                                                v1_i_o = T.axis.spatial(1, 0)
                                                v2_i_o = T.axis.spatial(1, 0)
                                                v3_i_o = T.axis.reduce(1, 0)
                                                T.reads(C_reindex_shared_dyn_warp_1[0, v1_o, v2_o, 0:32, 0:8], A_reindex_shared_dyn_warp_1[0, v1_o, v3_o, 0:32, 0:8], B_reindex_shared_dyn_warp_1[0, v3_o, v2_o, 0:32, 0:8])
                                                T.writes(C_reindex_shared_dyn_warp_1[0, v1_o, v2_o, 0:32, 0:8])
                                                A_1 = T.match_buffer(A_reindex_shared_dyn_warp_1[0, v1_o, v3_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                B_1 = T.match_buffer(B_reindex_shared_dyn_warp_1[0, v3_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                C_1 = T.match_buffer(C_reindex_shared_dyn_warp_1[0, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data, A_1.elem_offset + tx * 8, B_1.data, B_1.elem_offset + tx * 8, C_1.data, C_1.elem_offset + tx * 8, T.bool(False))
                                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data, A_1.elem_offset + tx * 8, B_1.data, B_1.elem_offset + tx * 8 + 4, C_1.data, C_1.elem_offset + tx * 8 + 4, T.bool(False))
                            for ax0_0, ax1_0 in T.grid(1, 2):
                                with T.block("C_reindex_shared.dyn_warp_o_1"):
                                    v0_o = T.axis.spatial(1, 0)
                                    v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused)
                                    v2_o = T.axis.spatial(8, ax2_0_2_1 * 2 + ax1_0)
                                    T.reads(C_reindex_shared_dyn_warp_1[v0_o, v1_o, v2_o, 0:32, 0:8])
                                    T.writes(C_reindex_shared_dyn_1[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                    C_warp = T.match_buffer(C_reindex_shared_dyn_warp_1[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=1)
                                    C_1 = T.match_buffer(C_reindex_shared_dyn_1[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_1_s0", "C_1_s1"), scope="shared.dyn", offset_factor=1)
                                    for tx in T.thread_binding(32, thread="threadIdx.x"):
                                        T.mma_store("float16", 16, 16, T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_1.strides[0] * 16, 2), C_warp.data, C_warp.elem_offset, C_1.strides[0])
        T.evaluate(0)

    @T.prim_func
    def gemm_0(A: T.Buffer((512, 128), "float16"), B: T.Buffer((128, 512), "float16"), C: T.Buffer((512, 512), "float16")):
        T.func_attr({"dlight.tensorcore_prenormlized": T.bool(True), "op_pattern": 0})
        # with T.block("root"):
        A_reindex_shared_dyn = T.alloc_buffer((1, 512, 128), "float16", scope="shared.dyn")
        B_reindex_shared_dyn = T.alloc_buffer((1, 128, 512), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_warp = T.alloc_buffer((1, 32, 8, 32, 8), "float16", scope="warp")
        B_reindex_shared_dyn_warp = T.alloc_buffer((1, 8, 32, 32, 8), "float16", scope="warp")
        C_reindex_shared_dyn = T.alloc_buffer((1, 512, 512), "float16", scope="shared.dyn")
        C_reindex_shared_dyn_warp = T.alloc_buffer((1, 32, 32, 32, 8), "float16", scope="warp")
        for ax0 in T.thread_binding(1, thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding(32, thread="blockIdx.y"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(1, thread="blockIdx.x"):
                    for ax1_0_2 in T.thread_binding(1, thread="threadIdx.y"):
                        for ax2_0_2 in T.thread_binding(4, thread="threadIdx.z"):
                            for ax1_0_3_init, ax2_0_3_init in T.grid(1, 8):
                                with T.block("gemm_o_init"):
                                    v0_o = T.axis.spatial(1, ax0)
                                    v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused + ax1_0_2 + ax1_0_3_init)
                                    v2_o = T.axis.spatial(32, ax2_0_2 * 8 + ax2_0_3_init)
                                    T.reads()
                                    T.writes(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8])
                                    with T.block("gemm_init_o"):
                                        v1_i_init_o = T.axis.spatial(1, 0)
                                        v2_i_init_o = T.axis.spatial(1, 0)
                                        T.reads()
                                        T.writes(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8])
                                        C_warp = T.match_buffer(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=1)
                                        for tx in T.thread_binding(32, thread="threadIdx.x"):
                                            T.mma_fill("float16", 8, C_warp.data, C_warp.elem_offset)
                            for ax3_0_0 in range(1):
                                for ax0_ax1_ax2_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                        for ax0_ax1_ax2_fused_2 in T.unroll(2, annotations={"pragma_unroll_explicit": 0}):
                                            for ax0_ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax0_ax1_ax2_fused_4 in T.vectorized(8):
                                                    with T.block("A_reindex_shared.dyn"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial(512, ax1_0_0_ax2_0_0_fused * 16 + (ax0_ax1_ax2_fused_0 * 2048 + ax0_ax1_ax2_fused_1 * 512 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) // 128)
                                                        v2 = T.axis.spatial(128, (ax0_ax1_ax2_fused_0 * 2048 + ax0_ax1_ax2_fused_1 * 512 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) % 128)
                                                        T.reads(A[v1, v2])
                                                        T.writes(A_reindex_shared_dyn[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "permuted_layout": 0})
                                                        A_reindex_shared_dyn[v0, v1, v2] = A[v1, v2]
                                for ax0_ax1_ax2_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                        for ax0_ax1_ax2_fused_2 in T.unroll(64, annotations={"pragma_unroll_explicit": 0}):
                                            for ax0_ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax0_ax1_ax2_fused_4 in T.vectorized(8):
                                                    with T.block("B_reindex_shared.dyn"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial(128, (ax0_ax1_ax2_fused_0 * 65536 + ax0_ax1_ax2_fused_1 * 16384 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) // 512)
                                                        v2 = T.axis.spatial(512, (ax0_ax1_ax2_fused_0 * 65536 + ax0_ax1_ax2_fused_1 * 16384 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) % 512)
                                                        T.reads(B[v1, v2])
                                                        T.writes(B_reindex_shared_dyn[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "permuted_layout": 0})
                                                        B_reindex_shared_dyn[v0, v1, v2] = B[v1, v2]
                                for ax3_0_1 in range(8):
                                    for ax0_0, ax1_0 in T.grid(1, 1):
                                        with T.block("A_reindex_shared.dyn_warp_o"):
                                            v0_o = T.axis.spatial(1, 0)
                                            v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused + ax0_0)
                                            v2_o = T.axis.spatial(8, ax3_0_1 + ax1_0)
                                            T.reads(A_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(A_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, 0:32, 0:8])
                                            T.block_attr({"permuted_layout": 0})
                                            warp = T.match_buffer(A_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                            shared = T.match_buffer(A_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                            for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + 8 * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * 16, 1), shared.strides[0] * (tx % 16) + 8 * (tx // 16))
                                    for ax0_0, ax1_0 in T.grid(1, 8):
                                        with T.block("B_reindex_shared.dyn_warp_o"):
                                            v0_o = T.axis.spatial(1, 0)
                                            v1_o = T.axis.spatial(8, ax3_0_1 + ax0_0)
                                            v2_o = T.axis.spatial(32, ax2_0_2 * 8 + ax1_0)
                                            T.reads(B_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(B_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, 0:32, 0:8])
                                            T.block_attr({"permuted_layout": 0})
                                            warp = T.match_buffer(B_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                            shared = T.match_buffer(B_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                            for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                T.ptx_ldmatrix("float16", T.bool(True), 4, ".b16", warp.data, warp.elem_offset + 8 * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * 16, 1), shared.strides[0] * (tx % 16) + 8 * (tx // 16))
                                    for ax1_0_3, ax2_0_3 in T.grid(1, 8):
                                        with T.block("gemm_o_update"):
                                            v0_o = T.axis.spatial(1, ax0)
                                            v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused + ax1_0_2 + ax1_0_3)
                                            v2_o = T.axis.spatial(32, ax2_0_2 * 8 + ax2_0_3)
                                            v3_o = T.axis.reduce(8, ax3_0_0 * 8 + ax3_0_1)
                                            T.reads(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8], A_reindex_shared_dyn_warp[0, v1_o, v3_o, 0:32, 0:8], B_reindex_shared_dyn_warp[0, v3_o, v2_o, 0:32, 0:8])
                                            T.writes(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8])
                                            with T.block("gemm_o"):
                                                v1_i_o = T.axis.spatial(1, 0)
                                                v2_i_o = T.axis.spatial(1, 0)
                                                v3_i_o = T.axis.reduce(1, 0)
                                                T.reads(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8], A_reindex_shared_dyn_warp[0, v1_o, v3_o, 0:32, 0:8], B_reindex_shared_dyn_warp[0, v3_o, v2_o, 0:32, 0:8])
                                                T.writes(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8])
                                                A_1 = T.match_buffer(A_reindex_shared_dyn_warp[0, v1_o, v3_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                B_1 = T.match_buffer(B_reindex_shared_dyn_warp[0, v3_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                C_1 = T.match_buffer(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data, A_1.elem_offset + tx * 8, B_1.data, B_1.elem_offset + tx * 8, C_1.data, C_1.elem_offset + tx * 8, T.bool(False))
                                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data, A_1.elem_offset + tx * 8, B_1.data, B_1.elem_offset + tx * 8 + 4, C_1.data, C_1.elem_offset + tx * 8 + 4, T.bool(False))
                            for ax0_0, ax1_0 in T.grid(1, 8):
                                with T.block("C_reindex_shared.dyn_warp_o"):
                                    v0_o = T.axis.spatial(1, 0)
                                    v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused)
                                    v2_o = T.axis.spatial(32, ax2_0_2 * 8 + ax1_0)
                                    T.reads(C_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, 0:32, 0:8])
                                    T.writes(C_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                    C_warp = T.match_buffer(C_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=1)
                                    C_1 = T.match_buffer(C_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_1_s0", "C_1_s1"), scope="shared.dyn", offset_factor=1)
                                    for tx in T.thread_binding(32, thread="threadIdx.x"):
                                        T.mma_store("float16", 16, 16, T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_1.strides[0] * 16, 2), C_warp.data, C_warp.elem_offset, C_1.strides[0])
                        for ax0_ax1_ax2_fused_0 in T.unroll(32, annotations={"pragma_unroll_explicit": 0}):
                            for ax0_ax1_ax2_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                for ax0_ax1_ax2_fused_2 in T.vectorized(8):
                                    with T.block("C_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(1, 0)
                                        v1 = T.axis.spatial(512, ax1_0_0_ax2_0_0_fused * 16 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2) // 512)
                                        v2 = T.axis.spatial(512, (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2) % 512)
                                        T.reads(C_reindex_shared_dyn[v0, v1, v2])
                                        T.writes(C[v1, v2])
                                        C[v1, v2] = C_reindex_shared_dyn[v0, v1, v2]

    @T.prim_func
    def gemm_1(A: T.Buffer((512, 512), "float16"), B: T.Buffer((512, 128), "float16"), C: T.Buffer((512, 128), "float16")):
        T.func_attr({"dlight.tensorcore_prenormlized": T.bool(True), "op_pattern": 0})
        # with T.block("root"):
        A_reindex_shared_dyn = T.alloc_buffer((1, 512, 512), "float16", scope="shared.dyn")
        B_reindex_shared_dyn = T.alloc_buffer((1, 512, 128), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_warp = T.alloc_buffer((1, 32, 32, 32, 8), "float16", scope="warp")
        B_reindex_shared_dyn_warp = T.alloc_buffer((1, 32, 8, 32, 8), "float16", scope="warp")
        C_reindex_shared_dyn = T.alloc_buffer((1, 512, 128), "float16", scope="shared.dyn")
        C_reindex_shared_dyn_warp = T.alloc_buffer((1, 32, 8, 32, 8), "float16", scope="warp")
        for ax0 in T.thread_binding(1, thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding(32, thread="blockIdx.y"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(1, thread="blockIdx.x"):
                    for ax1_0_2 in T.thread_binding(1, thread="threadIdx.y"):
                        for ax2_0_2 in T.thread_binding(4, thread="threadIdx.z"):
                            for ax1_0_3_init, ax2_0_3_init in T.grid(1, 2):
                                with T.block("gemm_o_init"):
                                    v0_o = T.axis.spatial(1, ax0)
                                    v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused + ax1_0_2 + ax1_0_3_init)
                                    v2_o = T.axis.spatial(8, ax2_0_2 * 2 + ax2_0_3_init)
                                    T.reads()
                                    T.writes(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8])
                                    with T.block("gemm_init_o"):
                                        v1_i_init_o = T.axis.spatial(1, 0)
                                        v2_i_init_o = T.axis.spatial(1, 0)
                                        T.reads()
                                        T.writes(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8])
                                        C_warp = T.match_buffer(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=1)
                                        for tx in T.thread_binding(32, thread="threadIdx.x"):
                                            T.mma_fill("float16", 8, C_warp.data, C_warp.elem_offset)
                            for ax3_0_0 in range(4):
                                for ax0_ax1_ax2_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                        for ax0_ax1_ax2_fused_2 in T.unroll(2, annotations={"pragma_unroll_explicit": 0}):
                                            for ax0_ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax0_ax1_ax2_fused_4 in T.vectorized(8):
                                                    with T.block("A_reindex_shared.dyn"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial(512, ax1_0_0_ax2_0_0_fused * 16 + (ax0_ax1_ax2_fused_0 * 2048 + ax0_ax1_ax2_fused_1 * 512 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) // 128)
                                                        v2 = T.axis.spatial(512, ax3_0_0 * 128 + (ax0_ax1_ax2_fused_0 * 2048 + ax0_ax1_ax2_fused_1 * 512 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) % 128)
                                                        T.reads(A[v1, v2])
                                                        T.writes(A_reindex_shared_dyn[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "permuted_layout": 0})
                                                        A_reindex_shared_dyn[v0, v1, v2] = A[v1, v2]
                                for ax0_ax1_ax2_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                        for ax0_ax1_ax2_fused_2 in T.unroll(16, annotations={"pragma_unroll_explicit": 0}):
                                            for ax0_ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax0_ax1_ax2_fused_4 in T.vectorized(8):
                                                    with T.block("B_reindex_shared.dyn"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial(512, ax3_0_0 * 128 + (ax0_ax1_ax2_fused_0 * 16384 + ax0_ax1_ax2_fused_1 * 4096 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) // 128)
                                                        v2 = T.axis.spatial(128, (ax0_ax1_ax2_fused_0 * 16384 + ax0_ax1_ax2_fused_1 * 4096 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) % 128)
                                                        T.reads(B[v1, v2])
                                                        T.writes(B_reindex_shared_dyn[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "permuted_layout": 0})
                                                        B_reindex_shared_dyn[v0, v1, v2] = B[v1, v2]
                                for ax3_0_1 in range(8):
                                    for ax0_0, ax1_0 in T.grid(1, 1):
                                        with T.block("A_reindex_shared.dyn_warp_o"):
                                            v0_o = T.axis.spatial(1, 0)
                                            v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused + ax0_0)
                                            v2_o = T.axis.spatial(32, ax3_0_0 * 8 + ax3_0_1 + ax1_0)
                                            T.reads(A_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(A_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, 0:32, 0:8])
                                            T.block_attr({"permuted_layout": 0})
                                            warp = T.match_buffer(A_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                            shared = T.match_buffer(A_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                            for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + 8 * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * 16, 1), shared.strides[0] * (tx % 16) + 8 * (tx // 16))
                                    for ax0_0, ax1_0 in T.grid(1, 2):
                                        with T.block("B_reindex_shared.dyn_warp_o"):
                                            v0_o = T.axis.spatial(1, 0)
                                            v1_o = T.axis.spatial(32, ax3_0_0 * 8 + ax3_0_1 + ax0_0)
                                            v2_o = T.axis.spatial(8, ax2_0_2 * 2 + ax1_0)
                                            T.reads(B_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(B_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, 0:32, 0:8])
                                            T.block_attr({"permuted_layout": 0})
                                            warp = T.match_buffer(B_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                            shared = T.match_buffer(B_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                            for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                T.ptx_ldmatrix("float16", T.bool(True), 4, ".b16", warp.data, warp.elem_offset + 8 * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * 16, 1), shared.strides[0] * (tx % 16) + 8 * (tx // 16))
                                    for ax1_0_3, ax2_0_3 in T.grid(1, 2):
                                        with T.block("gemm_o_update"):
                                            v0_o = T.axis.spatial(1, ax0)
                                            v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused + ax1_0_2 + ax1_0_3)
                                            v2_o = T.axis.spatial(8, ax2_0_2 * 2 + ax2_0_3)
                                            v3_o = T.axis.reduce(32, ax3_0_0 * 8 + ax3_0_1)
                                            T.reads(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8], A_reindex_shared_dyn_warp[0, v1_o, v3_o, 0:32, 0:8], B_reindex_shared_dyn_warp[0, v3_o, v2_o, 0:32, 0:8])
                                            T.writes(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8])
                                            with T.block("gemm_o"):
                                                v1_i_o = T.axis.spatial(1, 0)
                                                v2_i_o = T.axis.spatial(1, 0)
                                                v3_i_o = T.axis.reduce(1, 0)
                                                T.reads(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8], A_reindex_shared_dyn_warp[0, v1_o, v3_o, 0:32, 0:8], B_reindex_shared_dyn_warp[0, v3_o, v2_o, 0:32, 0:8])
                                                T.writes(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8])
                                                A_1 = T.match_buffer(A_reindex_shared_dyn_warp[0, v1_o, v3_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                B_1 = T.match_buffer(B_reindex_shared_dyn_warp[0, v3_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                C_1 = T.match_buffer(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data, A_1.elem_offset + tx * 8, B_1.data, B_1.elem_offset + tx * 8, C_1.data, C_1.elem_offset + tx * 8, T.bool(False))
                                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data, A_1.elem_offset + tx * 8, B_1.data, B_1.elem_offset + tx * 8 + 4, C_1.data, C_1.elem_offset + tx * 8 + 4, T.bool(False))
                            for ax0_0, ax1_0 in T.grid(1, 2):
                                with T.block("C_reindex_shared.dyn_warp_o"):
                                    v0_o = T.axis.spatial(1, 0)
                                    v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused)
                                    v2_o = T.axis.spatial(8, ax2_0_2 * 2 + ax1_0)
                                    T.reads(C_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, 0:32, 0:8])
                                    T.writes(C_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                    C_warp = T.match_buffer(C_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=1)
                                    C_1 = T.match_buffer(C_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_1_s0", "C_1_s1"), scope="shared.dyn", offset_factor=1)
                                    for tx in T.thread_binding(32, thread="threadIdx.x"):
                                        T.mma_store("float16", 16, 16, T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_1.strides[0] * 16, 2), C_warp.data, C_warp.elem_offset, C_1.strides[0])
                        for ax0_ax1_ax2_fused_0 in T.unroll(8, annotations={"pragma_unroll_explicit": 0}):
                            for ax0_ax1_ax2_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                for ax0_ax1_ax2_fused_2 in T.vectorized(8):
                                    with T.block("C_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(1, 0)
                                        v1 = T.axis.spatial(512, ax1_0_0_ax2_0_0_fused * 16 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2) // 128)
                                        v2 = T.axis.spatial(128, (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2) % 128)
                                        T.reads(C_reindex_shared_dyn[v0, v1, v2])
                                        T.writes(C[v1, v2])
                                        C[v1, v2] = C_reindex_shared_dyn[v0, v1, v2]

    @R.function
    def main(A: R.Tensor((512, 128), dtype="float16"), B: R.Tensor((128, 512), dtype="float16"), D: R.Tensor((512, 128), dtype="float16"), E: R.Tensor((512, 128), dtype="float16")) -> R.Tensor((512, 128), dtype="float16"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.fused_gemm_0_gemm_1, (A, B, D), out_sinfo=R.Tensor((512, 128), dtype="float16"))
            R.output(gv)
        return gv


@tvm.ir.transform.module_pass(opt_level=0, name="ReplaceRedundantVars")
class ReplaceRedundantVars:
    def transform_module(self, mod, ctx):
        # Get the specific function we want to transform
        func_name = "fused_gemm_0_gemm_1"
        func = None
        for gv, f in mod.functions.items():
            if gv.name_hint == func_name:
                func = f
                break
        
        if func is None:
            print(f"Function {func_name} not found in module")
            return mod
        
        # Extract the body of the function
        body = func.body
        
        # Create a variable mapping for the redundant variables
        var_mapping = {}
        
        # Find the original variables and create mappings
        def find_original_vars(stmt):
            if isinstance(stmt, tir.For):
                # Check if this is one of our target variables
                if stmt.loop_var.name == "ax1_0_0_ax2_0_0_fused":
                    # Store the original variable for later use
                    var_mapping["ax1_0_0_ax2_0_0_fused_1"] = stmt.loop_var
                elif stmt.loop_var.name == "ax1_0_2":
                    var_mapping["ax1_0_2_1"] = stmt.loop_var
                elif stmt.loop_var.name == "ax2_0_2":
                    var_mapping["ax2_0_2_1"] = stmt.loop_var
                
                # Recursively process the body
                find_original_vars(stmt.body)
            elif isinstance(stmt, tir.SeqStmt):
                for s in stmt.seq:
                    find_original_vars(s)
            elif isinstance(stmt, tir.Block):
                find_original_vars(stmt.body)
            elif isinstance(stmt, tir.BlockRealize):
                find_original_vars(stmt.block)
        
        # Find the original variables
        find_original_vars(body)
        
        # Replace redundant variables with original ones
        def replace_vars(stmt):
            if isinstance(stmt, tir.Var):
                # Check if this is one of our redundant variables
                if stmt.name in var_mapping:
                    return var_mapping[stmt.name]
                return stmt
            elif isinstance(stmt, tir.For):
                return tir.For(
                    stmt.loop_var,
                    replace_vars(stmt.min) if isinstance(stmt.min, (tir.Var, tir.Call, tir.Add, tir.Mul)) else stmt.min,
                    replace_vars(stmt.extent) if isinstance(stmt.extent, (tir.Var, tir.Call, tir.Add, tir.Mul)) else stmt.extent,
                    stmt.kind,
                    replace_vars(stmt.body),
                    stmt.thread_binding,
                    stmt.annotations
                )
            elif isinstance(stmt, tir.SeqStmt):
                return tir.SeqStmt([replace_vars(s) for s in stmt.seq])
            elif isinstance(stmt, tir.Block):
                return tir.Block(
                    stmt.iter_vars,
                    stmt.reads,
                    stmt.writes,
                    stmt.name_hint,
                    replace_vars(stmt.body),
                    replace_vars(stmt.init) if stmt.init else None,
                    stmt.alloc_buffers,
                    stmt.match_buffers,
                    stmt.annotations
                )
            elif isinstance(stmt, tir.BlockRealize):
                return tir.BlockRealize(
                    [replace_vars(v) if isinstance(v, (tir.Var, tir.Call, tir.Add, tir.Mul)) else v for v in stmt.iter_values],
                    replace_vars(stmt.predicate) if isinstance(stmt.predicate, (tir.Var, tir.Call, tir.Add, tir.Mul)) else stmt.predicate,
                    replace_vars(stmt.block)
                )
            elif isinstance(stmt, tir.Call):
                return tir.Call(
                    stmt.dtype,
                    stmt.op,
                    [replace_vars(arg) if isinstance(arg, (tir.Var, tir.Call, tir.Add, tir.Mul)) else arg for arg in stmt.args]
                )
            elif isinstance(stmt, tir.Add):
                return tir.Add(
                    replace_vars(stmt.a) if isinstance(stmt.a, (tir.Var, tir.Call, tir.Add, tir.Mul)) else stmt.a,
                    replace_vars(stmt.b) if isinstance(stmt.b, (tir.Var, tir.Call, tir.Add, tir.Mul)) else stmt.b
                )
            elif isinstance(stmt, tir.Mul):
                return tir.Mul(
                    replace_vars(stmt.a) if isinstance(stmt.a, (tir.Var, tir.Call, tir.Add, tir.Mul)) else stmt.a,
                    replace_vars(stmt.b) if isinstance(stmt.b, (tir.Var, tir.Call, tir.Add, tir.Mul)) else stmt.b
                )
            elif isinstance(stmt, tir.Evaluate):
                if isinstance(stmt.value, (tir.Var, tir.Call, tir.Add, tir.Mul)):
                    return tir.Evaluate(replace_vars(stmt.value))
                return stmt
            else:
                return stmt
        
        # Apply variable replacement
        new_body = replace_vars(body)
        
        # Create a new function with the updated body
        new_func = tir.PrimFunc(
            func.params,
            new_body,
            func.ret_type,
            func.buffer_map,
            func.attrs
        )
        
        # Update the module with the new function
        updated_mod = tvm.IRModule(mod.functions)
        for gv, f in mod.functions.items():
            if gv.name_hint == func_name:
                updated_mod[gv] = new_func
                break
        
        return updated_mod

# Example usage:
mod = Module
transformed_mod = ReplaceRedundantVars()(mod)
print(transformed_mod.script())