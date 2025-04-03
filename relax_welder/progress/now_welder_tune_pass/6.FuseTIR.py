
from hmac import new
from syslog import LOG_ALERT
from types import ModuleType
from numpy import block
from tvm import target
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
import tvm
from tvm import tir
from tvm.script.ir_builder.tir.ir import thread_binding
from tvm.tir import stmt_functor
from tvm import ir, transform, relax
import os

from tvm.tir.stmt import ForKind
from tvm.tir.transform.transform import TransformMmaBufferLayout
from collections import defaultdict


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



# 按照pre的顺序访问ast，并打印对应信息
def VisitAllNode():

    def _pre_visit(stmt):
        print(f"Pre Visiting node of type: {type(stmt)}")
        print(f"Pre Visiting node {stmt}")
        print("----------------------------------------------------------------------------------")
        return None
    
    def _post_visit(stmt):
        print(f"Post visiting node of type: {type(stmt)}")
        print(f"Post visiting node {stmt}")
        print("----------------------------------------------------------------------------------")
        return None

    def _ftransform(f, mod, ctx):
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(
                f.body,
                _pre_visit,
                _post_visit,
                # ["tir.Var"]
            )
        )
    
    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)



class ForCollector:
    thread_for_map = defaultdict(list)

# 建立一个thead_for_map, 后续可以用name 找到1个list，对应的是每个for node
def CollectForPass():
    # 使用栈进行优化
    valid_thread_tags = {
        "threadIdx.x", "threadIdx.y", "threadIdx.z",
        "blockIdx.x", "blockIdx.y", "blockIdx.z"
    }
    def _pre_visit(stmt):
        if hasattr(stmt, "thread_binding") and stmt.thread_binding is not None:
            thread_tag = stmt.thread_binding.thread_tag
            if thread_tag in valid_thread_tags:
                ForCollector.thread_for_map[thread_tag].append(stmt)
        return None

    def _ftransform(func, mod, ctx):
        return func.with_body(
            tvm.tir.stmt_functor.ir_transform(
                func.body,
                _pre_visit,
                None,
                ["tir.For"]
            )
        )
    
    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)

# 1. 找到primfunc2中 ax2_0_2(theadIdx.z) for node -> 对ax2_0_2 for node的index改为blockIdx.x.body.body[0]
# 2. 对primfunc1中 ax2_0_2 for node进行重建，加入primfunc2 ax2_0_2 for node中
# 可以使用threadbindings进行优化
def ParseFor():
    thread_for_map = ForCollector.thread_for_map


    def _post_visit(stmt):
        if isinstance(stmt, tir.For) and stmt == thread_for_map["threadIdx.z"][0]:
            return tvm.tir.For(
                loop_var=stmt.loop_var,
                min=stmt.min,
                extent=stmt.extent,
                kind=stmt.kind,
                body=tvm.tir.SeqStmt([*stmt.body, *thread_for_map["threadIdx.z"][3].body]),
                thread_binding=stmt.thread_binding,
                annotations=stmt.annotations
            )
        return stmt

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
# 删除primfunc2 ax0 for node下面的循环
def RemoveSpecificForNodePass():
    thread_for_map = ForCollector.thread_for_map
    
    def _post_visit(stmt):
        # Check if this is the target for node structure
        if isinstance(stmt, tvm.tir.For) and stmt == thread_for_map["blockIdx.z"][1]:
                return tvm.tir.Evaluate(tvm.tir.const(0, "int32"))
        
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


# 删除primfunc2 ax3_0_0 for node 的Block(A_reindex_shared.dyn_1)
def RemoveFor():
    thread_for_map = ForCollector.thread_for_map
    def _post_visit(stmt):
        if isinstance(stmt, tvm.tir.For) and stmt == thread_for_map["threadIdx.y"][4]:
            return tvm.tir.Evaluate(tvm.tir.const(0, "int32"))
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
            

# can replace all the buffer
# (A_reindex_shared_dyn_1 -> C_reindex_shared_dyn)
# (A_reindex_shared_dyn_warp_1 -> C_reindex_shared_dyn_warp)  
# (C_reindex_shared_dyn -> C_reindex_shared_dyn_1) (只在 C_reindex_shared.dyn)

def ReplaceBufferPass():
    # Dictionary to store the mapping from source buffer to target buffer
    buffer_map = {}
    target_buffers = {} #该map建立name -> buffer的结构
    name_to_buffer_map = {}

    def _pre_visit(stmt):
        nonlocal name_to_buffer_map
        if isinstance(stmt, tir.Block) and stmt.name_hint == "root":
            name_to_buffer_map = {buf.name: buf for buf in stmt.alloc_buffers}
        return None

    def _post_visit(stmt):
        nonlocal name_to_buffer_map, buffer_map
        if isinstance(stmt, tir.Block):
            # Replace buffer references in reads
            replace_buffer_name = ["A_reindex_shared_dyn_1", "A_reindex_shared_dyn_warp_1"]
            replace_Cbuffer = "C_reindex_shared_dyn"

            # 在block C_reindex_shared.dyn中对其进行修改
            if stmt.name_hint == "C_reindex_shared.dyn":
                buffer_map[replace_Cbuffer] = name_to_buffer_map[replace_Cbuffer + "_1"]
            else:
                if replace_Cbuffer in buffer_map:
                    del buffer_map[replace_Cbuffer] 


            for buf_name in replace_buffer_name:
                buffer_map[buf_name] = name_to_buffer_map["C" + buf_name[1:-2]]

            new_reads = []
            for read in stmt.reads:
                if read.buffer.name in buffer_map:
                    # Create a new BufferRegion with the target buffer
                    new_reads.append(tir.BufferRegion(buffer_map[read.buffer.name], read.region))
                else:
                    new_reads.append(read)
            
            # Replace buffer references in writes
            new_writes = []
            for write in stmt.writes:
                if write.buffer.name in buffer_map:
                    # Create a new BufferRegion with the target buffer
                    new_writes.append(tir.BufferRegion(buffer_map[write.buffer.name], write.region))
                else:
                    new_writes.append(write)
            
            
            # Create a new block with updated reads and writes if needed
            new_match_buffers = []
            for match in stmt.match_buffers:
                if match.source.buffer.name in buffer_map:
                    # Create a new BufferRegion with the target buffer
                    new_source = tir.BufferRegion(buffer_map[match.source.buffer.name], match.source.region)
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
        
        elif isinstance(stmt, tir.BufferLoad) or isinstance(stmt, tir.BufferStore):
            # Check if this BufferLoad uses C_reindex_shared_dyn buffer
            if stmt.buffer.name in buffer_map:
                # Create a new buffer with the same properties but different name
                new_buffer = buffer_map[stmt.buffer.name]
                # Create a new BufferLoad with the new buffer
                return tir.BufferLoad(new_buffer, stmt.indices, stmt.span)
        
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
        # print(f"Found {len(buffer_map)} buffers to replace")
        # for src, tgt in buffer_map.items():
        #     print(f"Replacing {src} with {tgt.name}")
            
        return result

    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)
            

# Substitue
# 使用thread bindings写一个等价的形式
# name_map 必须存在，用来记录loop_var的name，方便后面替换
def SubstituteAxis():
    thread_for_map = ForCollector.thread_for_map
    del thread_for_map["threadIdx.x"]
    name_map = {}
    vmap = {}

    for key in thread_for_map.keys():
        loop_var = thread_for_map[key][0].loop_var
        name_map[loop_var.name] = loop_var


    def _pre_visit(stmt):
        if isinstance(stmt, tvm.tir.Var) and stmt.name in name_map:
            if stmt != name_map[stmt.name]:
                vmap[stmt] = name_map[stmt.name]
        return None
    
    def _post_visit(stmt):
        if isinstance(stmt, tvm.tir.Var):
            return tvm.tir.stmt_functor.substitute(stmt, vmap)

        return stmt 
        
    def _ftransform(f, mod, ctx):
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(
                f.body,
                _pre_visit,
                _post_visit,
                ["tir.For", "tir.Var"]
            )
        )
    
    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)


# 删除root block中对global memory的alloc buffer
def DelCBuffer():
    def _post_visit(stmt):
        if stmt.name_hint == "root":
            new_alloc_buffer = [buf for buf in stmt.alloc_buffers if buf.name != "C_intermediate"]
            return tvm.tir.Block(
                iter_vars=stmt.iter_vars,
                reads=stmt.reads,
                writes=stmt.writes,
                alloc_buffers=new_alloc_buffer,
                match_buffers=stmt.match_buffers,
                name_hint=stmt.name_hint,
                init=stmt.init,
                body=stmt.body,
                annotations=stmt.annotations,
            )
    def _ftransform(f, mod, ctx):
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(
                f.body,
                None,
                _post_visit,
                ["tir.Block"]
            )
        )
    
    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)

# 将C_intermediate替换成C_intermediate_1
def LeaveBlock():
    target_buffer = None

    def _pre_visit(stmt):
        # nonlocal target_buffer
        # if isinstance(stmt, tir.Block) and stmt.name_hint == "root":
        #     for buffer in stmt.alloc_buffers:
        #         if buffer.name == "C_intermediate_1":
        #             target_buffer[buffer.name] = stmt
        #             print("C_intermediate_1 have been found")
        return None

    def _post_visit(stmt):
        nonlocal target_buffer
        if isinstance(stmt, tir.Block) and stmt.name_hint == "C_reindex_shared.dyn":
            # 只有当我们找到了C_intermediate_1 buffer才进行替换
            
            # 创建新的writes列表，替换引用C_intermediate的BufferRegion
            new_writes = []
            for write in stmt.writes:
                if write.buffer.name == "C_intermediate":
                    # 创建新的BufferRegion，使用C_intermediate_1 buffer但保留原始region
                    new_writes.append(tir.BufferRegion(
                        target_buffer,  # 新的buffer
                        write.region  # 保留原始region
                    ))
                    print(f"Replaced C_intermediate with C_intermediate_1 in writes")
                else:
                    new_writes.append(write)
            
            # 创建新的Block，只替换writes部分
            return tir.Block(
                iter_vars=stmt.iter_vars,
                reads=stmt.reads,
                writes=new_writes,  # 使用更新后的writes
                name_hint=stmt.name_hint,
                body=stmt.body,
                init=stmt.init,
                alloc_buffers=stmt.alloc_buffers,
                match_buffers=stmt.match_buffers,
                annotations=stmt.annotations,
            )

        if isinstance(stmt, tir.BufferStore):
            if stmt.buffer.name == "C_intermediate":
                # print(f"the buffer store's buffer is C_intermediate")
                return tir.BufferStore(
                    buffer=target_buffer,
                    value=stmt.value,
                    indices=stmt.indices,
                    span=stmt.span
                )
        return stmt


    def _ftransform(func, mod, ctx):
        nonlocal target_buffer
        for param in func.params:
            buf = func.buffer_map[param]
            if buf.name == "C_intermediate_1":
                target_buffer = buf
                print(f"found buffer")


        return func.with_body(
            tvm.tir.stmt_functor.ir_transform(
                func.body,
                _pre_visit,
                _post_visit,
                ["tir.Block", "tir.BufferStore"]
            )
        )

    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)


# Load the module
mod = MyModule

# Apply the pass


transformed_mod = CollectForPass()(mod)
write_mod(transformed_mod, log_path, "Basic")
transformed_mod = ParseFor()(transformed_mod)
write_mod(transformed_mod, log_path, "ParseFor")
transformed_mod = RemoveSpecificForNodePass()(transformed_mod)
write_mod(transformed_mod, log_path, "RemoveSpecificForNodePass")
transformed_mod = RemoveFor()(transformed_mod)
write_mod(transformed_mod, log_path, "RemoveFor")

transformed_mod = ReplaceBufferPass()(transformed_mod)
write_mod(transformed_mod, log_path, "ReplaceBufferPass")
transformed_mod = SubstituteAxis()(transformed_mod)
write_mod(transformed_mod, log_path, "SubstituteAxis")
transformed_mod = DelCBuffer()(transformed_mod)
write_mod(transformed_mod, log_path, "DelCBuffer")
transformed_mod = LeaveBlock()(transformed_mod)
write_mod(transformed_mod, log_path, "LeaveBlock")
# transformed_mod = VisitAllNode()(transformed_mod)
# transformed_mod = BlockInfo()(transformed_mod)


