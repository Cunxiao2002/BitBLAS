
from hmac import new
from syslog import LOG_ALERT
from types import ModuleType
from numpy import block
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
import tvm
from tvm import tir
from tvm.script.ir_builder.tir.ir import thread_binding
from tvm.tir import stmt_functor
from tvm import ir, transform, relax
import os

from tvm.tir.transform.transform import TransformMmaBufferLayout


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


class VmapCollector():
    vmap_collector = {}
    name_map = {}
    buffer_name_map = {}
    buffer_vmap = {}

    @staticmethod
    def get_vmap():
        return VmapCollector.vmap_collector

    @staticmethod
    def reset_vmap():
        VmapCollector.vmap_collector = {}


# 
def BufferNameMap():
    def _pre_visit(stmt):
        pass



# 创建VmapCollector中所需要的name_map
def CreateNameMap():
    var_targets = ["ax0", "ax1_0_0_ax2_0_0_fused", "ax1_0_1_ax2_0_1_fused", "ax1_0_2", "ax2_0_2"]
    # 改成thread bindings
    def _pre_visit(stmt):
        if stmt.name in var_targets:
            var_targets.remove(stmt.name)
            VmapCollector.name_map[stmt.name] = stmt
        return None
    
    def _ftransform(f, mod, ctx):
        return f.with_body(tvm.tir.stmt_functor.ir_transform(f.body, _pre_visit, None, ["tir.Var"]))

    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)

# 创建需要替换的vmap
def CreateVmap():
    var_targets = ["ax0_1", "ax1_0_0_ax2_0_0_fused_1", "ax1_0_1_ax2_0_1_fused_1", "ax1_0_2_1", "ax2_0_2_1"]

    def _pre_visit(stmt):
        if isinstance(stmt, tir.Var) and stmt.name in var_targets:
            # print(f"Processing var: {stmt.name} (id={id(stmt)})")
            VmapCollector.vmap_collector[stmt] = VmapCollector.name_map[stmt.name.replace("_1", "")]
        
    def _ftransform(f, mod, ctx):
        return f.with_body(tvm.tir.stmt_functor.ir_transform(f.body, _pre_visit, None, ["tir.Var"]))

    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)

# 使用创建好的vmap进行substitute
def SubstituteVmap():    
    var_targets = ["ax0_1", "ax1_0_0_ax2_0_0_fused_1", "ax1_0_1_ax2_0_1_fused_1", "ax1_0_2_1", "ax2_0_2_1"]
    def _post_visit(stmt):
        if isinstance(stmt, tir.Var) and stmt.name in var_targets:
            # print(f"after substitute, Var is {stmt}")
            vmap = VmapCollector.get_vmap()
            return tvm.tir.stmt_functor.substitute(stmt, vmap)
        return stmt
    
    def _ftransform(f, mod, ctx):
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(
                f.body,
                None,
                _post_visit,
                # ["tir.Var"]
            )
        )
    
    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)



def VisitAllNode():

    def _pre_visit(stmt):
        print(f"Visiting node of type: {type(stmt)}")
        print(f"Visiting node {stmt}")
        print("----------------------------------------------------------------------------------")
        return None
    
    def _ftransform(f, mod, ctx):
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(
                f.body,
                _pre_visit,
                None,
                # ["tir.Var"]
            )
        )

    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)



class ForCollector:
    collected_for = None

    @staticmethod
    def reset():
        ForCollector.collected_for = None

    @staticmethod
    def get_for():
        return ForCollector.collected_for

# 收集For node相关的信息
def CollectForPass():
    for_name = "ax2_0_2"
    first_found = False
    def _pre_visit(stmt):
        if isinstance(stmt, tir.For) and stmt.loop_var.name == for_name and not first_found :
            # print(stmt)
            ForCollector.collected_for = stmt
        
        return None

    def _ftransform(func, mod, ctx):
        return func.with_body(
            tvm.tir.stmt_functor.ir_transform(
                func.body,
                _pre_visit,
                None,  # 不需要post_visit
                ["tir.For"]
            )
        )
    
    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)

def ParseFor():
    target_loop_var = "ax2_0_2"

    def _pre_visit(stmt):
        if isinstance(stmt, tir.For):
            return None            
        return stmt

    def _post_visit(stmt):
        if isinstance(stmt, tir.For) and stmt.loop_var.name == target_loop_var:
            collected_for = ForCollector.get_for()
            # print(f"collected_for is {collected_for}")
            if collected_for is not None:
                return tvm.tir.For(
                    loop_var=stmt.loop_var,
                    min=stmt.min,
                    extent=stmt.extent,
                    kind=stmt.kind,
                    body=tvm.tir.SeqStmt([stmt.body[0], stmt.body[1], stmt.body[2], collected_for.body[0], collected_for.body[1], collected_for.body[2]]),
                    thread_binding=stmt.thread_binding,
                    annotations=stmt.annotations
                )

    def _ftransform(func, mod, ctx):
        return func.with_body(
            tvm.tir.stmt_functor.ir_transform(
                func.body,
                None,
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
    temp = 0

    
    def _pre_visit(stmt):
        return None
    
    def _post_visit(stmt):
        nonlocal first_occurrence_found, second_occurrence_removed, temp
        
        # Check if this is the target for node structure
        if isinstance(stmt, tvm.tir.For) and stmt.loop_var.name == "ax0" and not second_occurrence_removed:
            temp += 1
            if(temp == 2):
                return tvm.tir.Evaluate(tvm.tir.const(0, "int32"))
        
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


def RemoveFor():
    temp_1 = 0
    temp_2 = 0
    def _post_visit(stmt):
        nonlocal temp_1, temp_2
        if stmt.loop_var.name == "ax1_0_3_init":
            temp_1 += 1
            if temp_1 == 2:
                return tvm.tir.Evaluate(tvm.tir.const(0, "int32"))
        if stmt.loop_var.name == "ax3_0_0":
            temp_2 += 1
            if temp_2 == 2:
                new_body = list(stmt.body)
                new_body.pop(0)
                new_seq = tvm.tir.SeqStmt(new_body)
                new_for = tvm.tir.For(
                    loop_var=stmt.loop_var,
                    min=stmt.min,
                    extent=stmt.extent,
                    kind=stmt.kind,
                    body=new_seq,
                    thread_binding=stmt.thread_binding,
                    annotations=stmt.annotations,
                    span=stmt.span
                )
                return new_for
        return stmt

    def _ftransform(f, mod, ctx):
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(
                f.body,
                None,
                _post_visit,
                ["tir.For"]
            )
        )
    
    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)




# can replace all the buffer(A_reindex_shared_dyn_1 -> C_reindex_shared_dyn)
# buffer(A_reindex_shared_dyn_warp_1 -> C_reindex_shared_dyn_warp)  
# maybe should change the name index to thread_bingding index
def ReplaceBufferPass():
    # Dictionary to store the mapping from source buffer to target buffer
    buffer_map = {}
    target_buffers = {}

    def _pre_visit(stmt):
        if isinstance(stmt, tir.Block):
            # First, find the target buffers in reads or writes
            if "C_reindex_shared_dyn" not in target_buffers:
                for read in stmt.reads:
                    if read.buffer.name == "C_reindex_shared_dyn":
                        target_buffers["C_reindex_shared_dyn"] = read.buffer
                        break
                if "C_reindex_shared_dyn" not in target_buffers:
                    for write in stmt.writes:
                        if write.buffer.name == "C_reindex_shared_dyn":
                            target_buffers["C_reindex_shared_dyn"] = write.buffer
                            break
            
            if "C_reindex_shared_dyn_warp" not in target_buffers:
                for read in stmt.reads:
                    if read.buffer.name == "C_reindex_shared_dyn_warp":
                        target_buffers["C_reindex_shared_dyn_warp"] = read.buffer
                        break
                if "C_reindex_shared_dyn_warp" not in target_buffers:
                    for write in stmt.writes:
                        if write.buffer.name == "C_reindex_shared_dyn_warp":
                            target_buffers["C_reindex_shared_dyn_warp"] = write.buffer
                            break
            
            # Then, find the source buffers and map them to the targets
            for read in stmt.reads:
                if read.buffer.name == "A_reindex_shared_dyn_1":
                    if "C_reindex_shared_dyn" in target_buffers:
                        buffer_map[read.buffer] = target_buffers["C_reindex_shared_dyn"]
                elif read.buffer.name == "A_reindex_shared_dyn_warp_1":
                    if "C_reindex_shared_dyn_warp" in target_buffers:
                        buffer_map[read.buffer] = target_buffers["C_reindex_shared_dyn_warp"]
            
            for write in stmt.writes:
                if write.buffer.name == "A_reindex_shared_dyn_1":
                    if "C_reindex_shared_dyn" in target_buffers:
                        buffer_map[write.buffer] = target_buffers["C_reindex_shared_dyn"]
                elif write.buffer.name == "A_reindex_shared_dyn_warp_1":
                    if "C_reindex_shared_dyn_warp" in target_buffers:
                        buffer_map[write.buffer] = target_buffers["C_reindex_shared_dyn_warp"]
        return None

    def _post_visit(stmt):
        if isinstance(stmt, tir.Block):
            # Replace buffer references in reads
            new_reads = []
            for read in stmt.reads:
                if read.buffer in buffer_map:
                    # Create a new BufferRegion with the target buffer
                    new_reads.append(tir.BufferRegion(buffer_map[read.buffer], read.region))
                else:
                    new_reads.append(read)
            
            # Replace buffer references in writes
            new_writes = []
            for write in stmt.writes:
                if write.buffer in buffer_map:
                    # Create a new BufferRegion with the target buffer
                    new_writes.append(tir.BufferRegion(buffer_map[write.buffer], write.region))
                else:
                    new_writes.append(write)
            
            
            # Create a new block with updated reads and writes if needed
            new_match_buffers = []
            for match in stmt.match_buffers:
                if match.source.buffer in buffer_map:
                    # Create a new BufferRegion with the target buffer
                    new_source = tir.BufferRegion(buffer_map[match.source.buffer], match.source.region)
                    # Create a new MatchBufferRegion with the new source
                    new_match_buffers.append(tir.MatchBufferRegion(match.buffer, new_source))
                else:
                    new_match_buffers.append(match)
            
            # Create a new block with updated reads, writes, and match_buffers if needed
            if new_reads != stmt.reads or new_writes != stmt.writes or new_match_buffers != stmt.match_buffers:
                return tir.Block(
                    stmt.iter_vars,
                    new_reads,
                    new_writes,
                    stmt.name_hint,
                    stmt.body,
                    stmt.init,
                    stmt.alloc_buffers,
                    new_match_buffers,  # Use the new match_buffers
                    stmt.annotations
                )
        
        return stmt

    def _ftransform(f, mod, ctx):
        # First perform the transform to populate the buffer_map
        result = f.with_body(tvm.tir.stmt_functor.ir_transform(
            f.body,
            _pre_visit,
            _post_visit,
            ["tir.Block", "tir.BufferLoad", "tir.BufferStore"]
        ))
        
        # Print the buffer mapping for debugging after the transform
        print(f"Found {len(buffer_map)} buffers to replace")
        for src, tgt in buffer_map.items():
            print(f"Replacing {src.name} with {tgt.name}")
            
        return result

    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)
            


def BlockInfo():
    def _pre_visit(stmt):
        print(f"the Block is\n {stmt}")
        print(f"the BLock's iter_values is {stmt.iter_vars}")
        print(f"the Block's reads is {stmt.reads}")
        print(f"the Block's writes is {stmt.writes}")
        print(f"the Block's name_hint is\n {stmt.name_hint}")
        print(f"the Block's body is\n {stmt.body}")
        print(f"the Block's init is {stmt.init}")
        print(f"the Block's alloc_buffers is {stmt.alloc_buffers}")
        print(f"the Block's match_buffers is {stmt.match_buffers}")
        print(f"the Block's annotations is {stmt.annotations}")
        print("---------------------------------------------------------------------------------------------------------------------------------")
        return None

    def _ftransform(f, mod, ctx):
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(
                f.body,
                _pre_visit,
                None,
                ["tir.Block"]
            )
        )
    
    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)

# Load the module
mod = MyModule

# Apply the pass


transformed_mod = CollectForPass()(mod)
write_mod(transformed_mod, log_path, "Basic")
# transform_mod = SubstituteVmap()(transformed_mod)
# write_mod(transformed_mod, log_path, "SubstituteVmap")
transformed_mod = ParseFor()(transformed_mod)
write_mod(transformed_mod, log_path, "ParseFor")
transformed_mod = RemoveSpecificForNodePass()(transformed_mod)
write_mod(transformed_mod, log_path, "RemoveSpecificForNodePass")
transformed_mod = RemoveFor()(transformed_mod)
write_mod(transformed_mod, log_path, "RemoveFor")
# transformed_mod = VisitAllNode()(transformed_mod)
# transformed_mod = BlockInfo()(transformed_mod)
transformed_mod = ReplaceBufferPass()(transformed_mod)
write_mod(transformed_mod, log_path, "ReplaceBufferPass")

# transformed_mod = CreateNameMap()(transformed_mod)
# transformed_mod = CreateVmap()(transformed_mod)
# transformed_mod = SubstituteVmap()(transformed_mod)
# write_mod(transformed_mod, log_path, "SubstituteVmap")


print(f"Vampcollector.name_map is {VmapCollector.name_map}")
print(f"VmapCollector.vmap is {VmapCollector.get_vmap()}")
# for var, mapped_var in VmapCollector.get_vmap().items():
#     print(f"Key[id={id(var)}]: {type(var).__name__} (Name: {var.name}), "
#           f"Value[id={id(mapped_var)}]: {type(mapped_var).__name__} "
#           f"(Name: {mapped_var.name if hasattr(mapped_var, 'name') else 'N/A'})")


# print(transformed_mod.script())
# print(f"the ForCollector.collected_for is \n{ForCollector.collected_for}")