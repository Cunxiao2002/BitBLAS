from hmac import new
from types import ModuleType
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
import tvm
from tvm import tir
from tvm.script.ir_builder.tir.ir import thread_binding
from tvm.tir import stmt_functor
from tvm import ir, transform, relax
import os


fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# get current file path
log_path = os.path.dirname(os.path.abspath(__file__)) + "/progress/" + fname

count = 0

def write_code(code, path, fname):
    global count
    fname = str(count) + "." + fname
    count += 1
    if not os.path.exists(path):
        os.makedirs(path)
    fname = os.path.join(path, fname)
    with open(fname, "w") as f:
        f.write(code)


def write_sch(sch, path, fname):
    py_fname = fname + ".py"
    write_code(sch.mod["main"].script(), path, py_fname)
    cu_fname = fname + ".cu"
    write_code(sch.mod.astext(), path, cu_fname)


def write_mod(mod, path, fname):
    py_fname = fname + ".py"
    write_code(mod.script(show_meta=False), path, py_fname)
    cu_fname = fname + ".cu"
    write_code(mod.astext(show_meta_data=False), path, cu_fname)


@I.ir_module
class MyModule:
    @T.prim_func(private=True)
    def fused_gemm_0_gemm_1(A: T.Buffer((T.int64(512), T.int64(128)), "float16"), B: T.Buffer((T.int64(128), T.int64(512)), "float16"), D: T.Buffer((T.int64(512), T.int64(128)), "float16"), C_intermediate_1: T.Buffer((T.int64(512), T.int64(128)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
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
                        for ax0_ax1_ax2_fused_0 in T.unroll(32, annotations={"pragma_unroll_explicit": 0}):
                            for ax0_ax1_ax2_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                for ax0_ax1_ax2_fused_2 in T.vectorized(8):
                                    with T.block("C_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(1, 0)
                                        v1 = T.axis.spatial(512, ax1_0_0_ax2_0_0_fused * 16 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2) // 512)
                                        v2 = T.axis.spatial(512, (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2) % 512)
                                        T.reads(C_reindex_shared_dyn[v0, v1, v2])
                                        T.writes(C_intermediate[v1, v2])
                                        C_intermediate[v1, v2] = C_reindex_shared_dyn[v0, v1, v2]
        for ax0 in T.thread_binding(1, thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding(32, thread="blockIdx.y"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(1, thread="blockIdx.x"):
                    for ax1_0_2 in T.thread_binding(1, thread="threadIdx.y"):
                        for ax2_0_2 in T.thread_binding(4, thread="threadIdx.z"):
                            for ax1_0_3_init, ax2_0_3_init in T.grid(1, 2):
                                with T.block("gemm_o_init_1"):
                                    v0_o = T.axis.spatial(1, ax0)
                                    v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused + ax1_0_2 + ax1_0_3_init)
                                    v2_o = T.axis.spatial(8, ax2_0_2 * 2 + ax2_0_3_init)
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
                                                        v1 = T.axis.spatial(512, ax1_0_0_ax2_0_0_fused * 16 + (ax0_ax1_ax2_fused_0 * 2048 + ax0_ax1_ax2_fused_1 * 512 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) // 128)
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
                                            v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused + ax0_0)
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
                                            v2_o = T.axis.spatial(8, ax2_0_2 * 2 + ax1_0)
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
                                            v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused + ax1_0_2 + ax1_0_3)
                                            v2_o = T.axis.spatial(8, ax2_0_2 * 2 + ax2_0_3)
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
                                    v2_o = T.axis.spatial(8, ax2_0_2 * 2 + ax1_0)
                                    T.reads(C_reindex_shared_dyn_warp_1[v0_o, v1_o, v2_o, 0:32, 0:8])
                                    T.writes(C_reindex_shared_dyn_1[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                    C_warp = T.match_buffer(C_reindex_shared_dyn_warp_1[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=1)
                                    C_1 = T.match_buffer(C_reindex_shared_dyn_1[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_1_s0", "C_1_s1"), scope="shared.dyn", offset_factor=1)
                                    for tx in T.thread_binding(32, thread="threadIdx.x"):
                                        T.mma_store("float16", 16, 16, T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_1.strides[0] * 16, 2), C_warp.data, C_warp.elem_offset, C_1.strides[0])
                        for ax0_ax1_ax2_fused_0 in T.unroll(8, annotations={"pragma_unroll_explicit": 0}):
                            for ax0_ax1_ax2_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                for ax0_ax1_ax2_fused_2 in T.vectorized(8):
                                    with T.block("C_reindex_shared.dyn_1"):
                                        v0 = T.axis.spatial(1, 0)
                                        v1 = T.axis.spatial(512, ax1_0_0_ax2_0_0_fused * 16 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2) // 128)
                                        v2 = T.axis.spatial(128, (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2) % 128)
                                        T.reads(C_reindex_shared_dyn_1[v0, v1, v2])
                                        T.writes(C_intermediate_1[v1, v2])
                                        C_intermediate_1[v1, v2] = C_reindex_shared_dyn_1[v0, v1, v2]


    @R.function
    def main(A: R.Tensor((512, 128), dtype="float16"), B: R.Tensor((128, 512), dtype="float16"), D: R.Tensor((512, 128), dtype="float16"), E: R.Tensor((512, 128), dtype="float16")) -> R.Tensor((512, 128), dtype="float16"):
        cls = MyModule
        with R.dataflow():
            gv = R.call_tir(cls.fused_gemm_0_gemm_1, (A, B, D), out_sinfo=R.Tensor((512, 128), dtype="float16"))
            R.output(gv)
        return gv


# (issue)用于将primfunc2的body移动到primfunc1 loop下面
def MoveInitLoopPass():
    extracted_loop = None
    target_loop_found = False
    
    def _pre_visit(stmt):
        nonlocal extracted_loop, target_loop_found
        
        if isinstance(stmt, tvm.tir.For) and stmt.loop_var.name == "ax2_0_2" and extracted_loop is None:
            # Look for the nested grid loop with ax1_0_3_init and ax2_0_3_init
            # print(f"the pre stmt is \n{stmt}")
            # print("------------------------------------------------------")
            # print(f"stmt.body is \n{stmt.body}")
            # print("------------------------------------------------------")
            # print(f"type of stmt.body is {type(stmt.body)}")
            # print("------------------------------------------------------")
            # print(f"the type of body's first stmt is {type(stmt.body[0])}")
            # print("------------------------------------------------------")
            # print(f"the body's first stmt is \n{stmt.body[0]}")
            # print("------------------------------------------------------")

            # print("------------------------------------------------------")
            # print(f"stmt.body is of type: {type(stmt.body)}")
            # print("------------------------------------------------------")
            # if hasattr(stmt.body, "__len__"):
            #     print(f"stmt.body has {len(stmt.body)} elements")
            #     for i, s in enumerate(stmt.body):
            #         print(f"Element {i} is of type: {type(s)}")
            #         if hasattr(s, "name_hint"):
            #             print(f"Element {i} name_hint: {s.name_hint}")
            #         else:
            #             print(f"Element {i} does not have name_hint attribute")
            #         print(f"Element {i}: {s}")
            #         print("------------------------------------------------------")

            for_body = stmt.body[0].body.body.block
            # print("------------------------------------------------------")
            # print(f"the type of for_body'body is {type(for_body.body)}")
            # print("------------------------------------------------------")
            # print(f"for_body‘s is \n{for_body.body.body.block}\n")
            # print("------------------------------------------------------")
            # print(f"the for_body's type {type(for_body)}")
            # print("------------------------------------------------------")
            if isinstance(for_body, tvm.tir.Block) and for_body.name_hint == "gemm_o_init_1":
                # for_body = for_body.body
                # print(f"for body is\n {for_body}")
                # print("------------------------------------------------------")
                extracted_loop = stmt.body
                # print(f"extracted_loop is \n {extracted_loop}")
                # print("------------------------------------------------------")
                return None
            
            if isinstance(for_body, tvm.tir.For) and for_body.loop_var.name == "ax1_0_3_init":
                # print(f"the for_body is \n{for_body}")
                # Found the loop we want to extract
                extracted_loop = for_body
                # Return None to continue traversal
                # print("------------------------------------------------------")
                # print(f"extracted_loop is \n {extracted_loop}")
                # print("------------------------------------------------------")
                return None
        
        if isinstance(stmt, tvm.tir.For) and stmt.loop_var.name == "ax2_0_2" and extracted_loop is not None and not target_loop_found:
            target_loop_found = True
            return None

        return None
    
    def _post_visit(stmt):
        nonlocal extracted_loop, target_loop_found
        # print(f"type of post stmt: {type(stmt)}")
        # print(f"target_loop_found is {target_loop_found}\n")
        if isinstance(stmt, tvm.tir.For) and stmt.loop_var.name == "ax2_0_2" and extracted_loop is not None and target_loop_found:
            block_detect = stmt.body[0].body.body.block
            # print("------------------------------------------------------")
            # print(f'block_detect is {block_detect}')
            # print("------------------------------------------------------")
            if isinstance(block_detect, tvm.tir.Block) and block_detect.name_hint == "gemm_o_init":
                # print("------------------------------------------------------")
                # print(f"post_stmt's body is {stmt.body}\n")
                # print("------------------------------------------------------")
                new_body = tvm.tir.SeqStmt([stmt.body, extracted_loop])
                # print("------------------------------------------------------")
                # print(f"new_body is \n{new_body}")
                # print("------------------------------------------------------")
                
                # print(f"stmt.loop_var is {stmt.loop_var}")
                new_stmt = tvm.tir.For(
                    loop_var=stmt.loop_var,
                    min=stmt.min,
                    extent=stmt.extent,
                    kind=stmt.kind,
                    body=new_body,
                    thread_binding=stmt.thread_binding,
                    annotations=stmt.annotations
                )

                # print("------------------------------------------------------")
                # print(f"new_stmt is \n{new_stmt}")
                # print("------------------------------------------------------")
                
                target_loop_found = False
                
                return new_stmt
        
        return stmt
    
    def _ftransform(f, mod, ctx):
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(
                f.body,
                _pre_visit,
                _post_visit,
                ["tir.For", "tir.Block"]
            )
        )
    
    # Return a pass that applies the transformation
    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)


# 
def ExtractAndSubstituteLoopVarsPass():
    """
    A pass that extracts loop variables from two different sections of a TIR function
    and substitutes the second set with the first set using the substitute method.
    """
    # Variables to store the extracted loop vars
    loop_vars1 = {}
    loop_vars2 = {}
    first_section_found = False
    second_section_found = False
    
    def _pre_visit(stmt):
        nonlocal loop_vars1, loop_vars2, first_section_found, second_section_found
        
        # Extract loop vars from the first section
        if isinstance(stmt, tvm.tir.For) and stmt.loop_var.name == "ax0" and not first_section_found:
            # if tvm.tir.all(stmt.thread_binding != 0, stmt.thread_binding.thread_tag == "blockIdx.z"):
            if stmt.thread_binding is not None:
                if stmt.thread_binding.thread_tag == "blockIdx.z":
                    first_section_found = True
                    loop_vars1["ax0"] = stmt.loop_var
                    
                    # Extract nested loop vars
                    body = stmt.body
                    if isinstance(body, tvm.tir.For) and body.loop_var.name == "ax1_0_0_ax2_0_0_fused":
                        loop_vars1["ax1_0_0_ax2_0_0_fused"] = body.loop_var
                        
                        body = body.body
                        if isinstance(body, tvm.tir.For) and body.loop_var.name == "ax1_0_1_ax2_0_1_fused":
                            loop_vars1["ax1_0_1_ax2_0_1_fused"] = body.loop_var
                            
                            body = body.body
                            if isinstance(body, tvm.tir.For) and body.loop_var.name == "ax1_0_2":
                                loop_vars1["ax1_0_2"] = body.loop_var
                                
                                body = body.body
                                if isinstance(body, tvm.tir.For) and body.loop_var.name == "ax2_0_2":
                                    loop_vars1["ax2_0_2"] = body.loop_var
            
        # Extract loop vars from the second section
        elif isinstance(stmt, tvm.tir.For) and stmt.loop_var.name == "ax0" and first_section_found and not second_section_found:
            # if (stmt.thread_binding and stmt.thread_binding.thread_tag == "blockIdx.z"):
            if stmt.thread_binding is not None:
                if stmt.thread_binding.thread_tag == "blockIdx.z":
                    second_section_found = True
                    loop_vars2["ax0"] = stmt.loop_var
                    
                    # Extract nested loop vars
                    body = stmt.body
                    if isinstance(body, tvm.tir.For) and body.loop_var.name == "ax1_0_0_ax2_0_0_fused":
                        loop_vars2["ax1_0_0_ax2_0_0_fused"] = body.loop_var
                        
                        body = body.body
                        if isinstance(body, tvm.tir.For) and body.loop_var.name == "ax1_0_1_ax2_0_1_fused":
                            loop_vars2["ax1_0_1_ax2_0_1_fused"] = body.loop_var
                            
                            body = body.body
                            if isinstance(body, tvm.tir.For) and body.loop_var.name == "ax1_0_2":
                                loop_vars2["ax1_0_2"] = body.loop_var
                                
                                body = body.body
                                if isinstance(body, tvm.tir.For) and body.loop_var.name == "ax2_0_2":
                                    loop_vars2["ax2_0_2"] = body.loop_var
        
        return None
    
    def _post_visit(stmt):
        nonlocal loop_vars1, loop_vars2, first_section_found, second_section_found
        
        # If we've found both sections and have all loop vars, perform substitution
        if first_section_found and second_section_found and loop_vars1 and loop_vars2:
            # Create variable mapping
            vmap = {}
            for name, var in loop_vars2.items():
                if name in loop_vars1:
                    vmap[var] = loop_vars1[name]
                    
            
            # Apply substitution
            # print("------------------------------------------------------")
            # print(f"type of stmt is {type(stmt)}\n")
            # print("------------------------------------------------------")
            # print(f"stmt is \n{stmt}")
            # print("------------------------------------------------------")
            # print(f"vmap is \n{vmap}")
            # print("------------------------------------------------------")
            # new_stmt = tvm.tir.stmt_functor.substitute(stmt, vmap)
            # print("------------------------------------------------------")
            # print(f"new_stmt is \n{new_stmt}")
            # print("------------------------------------------------------")
            return tvm.tir.stmt_functor.substitute(stmt, vmap)
        
        return stmt
    
    def _ftransform(f, mod, ctx):
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(
                f.body,
                _pre_visit,
                _post_visit,
                ["tir.For"]
            )
        )
    
    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)

# this pass can remove the prim_func2 from fused_primfunc
def RemoveSpecificForNodePass():
    """
    A pass that removes a specific for node structure with thread bindings:
    for ax0 in T.thread_binding(1, thread="blockIdx.z"):
        for ax1_0_0_ax2_0_0_fused in T.thread_binding(32, thread="blockIdx.y"):
            for ax1_0_1_ax2_0_1_fused in T.thread_binding(1, thread="blockIdx.x"):
                for ax1_0_2 in T.thread_binding(1, thread="threadIdx.y"):
                    for ax2_0_2 in T.thread_binding(4, thread="threadIdx.z"):
                        ...
    """
    first_occurrence_found = False
    second_occurrence_removed = False

    
    def _pre_visit(stmt):
        return None
    
    def _post_visit(stmt):
        nonlocal first_occurrence_found, second_occurrence_removed
        
        # Check if this is the target for node structure
        if isinstance(stmt, tvm.tir.For) and stmt.loop_var.name == "ax0" and not second_occurrence_removed:
            if (stmt.thread_binding is not None and 
                stmt.thread_binding.thread_tag == "blockIdx.z"):
                
                body = stmt.body
                # print(f"the body is \n {body}")
                if (isinstance(body, tvm.tir.For) and 
                    body.loop_var.name == "ax1_0_0_ax2_0_0_fused" and 
                    body.thread_binding is not None and 
                    body.thread_binding.thread_tag == "blockIdx.y"):
                    
                    body = body.body
                    # print(f"the body is \n {body}")
                    if (isinstance(body, tvm.tir.For) and 
                        body.loop_var.name == "ax1_0_1_ax2_0_1_fused" and 
                        body.thread_binding is not None and 
                        body.thread_binding.thread_tag == "blockIdx.x"):
                        
                        body = body.body
                        if (isinstance(body, tvm.tir.For) and 
                            body.loop_var.name == "ax1_0_2" and 
                            body.thread_binding is not None and 
                            body.thread_binding.thread_tag == "threadIdx.y"):
                            
                            body = body.body[0]
                            # print(f"the body is \n {body}")
                            # print("--------------------------------")
                            # print(f"the type of the body{type(body)}")
                            # print("--------------------------------")
                            if (isinstance(body, tvm.tir.For) and 
                                body.loop_var.name == "ax2_0_2" and 
                                body.thread_binding is not None and 
                                body.thread_binding.thread_tag == "threadIdx.z"):
                                if not first_occurrence_found:
                                    first_occurrence_found = True
                                    return stmt
                                # We found the target structure, mark it and return an empty statement
                                else:
                                    second_occurrence_removed = True
                                    return tvm.tir.Evaluate(tvm.tir.const(0, "int32"))
                                # print(f"the body is \n {body}")
                                # print("--------------------------------")
                                # print(f"substitute stmt is \n{stmt}")
                                # print("--------------------------------")
        
        return stmt
    
    def _ftransform(f, mod, ctx):
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(
                f.body,
                _pre_visit,
                _post_visit,
                ["tir.For"]
            )
        )
    
    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)

# Load the module
mod = MyModule
# MoveInitLoopPass_1 = MoveInitLoopPass()
# Apply the pass
# transformed_mod = MoveInitLoopPass()(mod)
transformed_mod = ExtractAndSubstituteLoopVarsPass()(mod)
# transformed_mod = RemoveSpecificForNodePass()(mod)
print(transformed_mod.script())