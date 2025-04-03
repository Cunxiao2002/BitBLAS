from ast import stmt
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
from tvm.tir import buffer, stmt_functor
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
class Module:
    @T.prim_func(private=True)
    def fused_gemm_0_gemm_1_gemm_2(A: T.Buffer((T.int64(512), T.int64(128)), "float16"), B: T.Buffer((T.int64(128), T.int64(512)), "float16"), D: T.Buffer((T.int64(512), T.int64(128)), "float16"), F: T.Buffer((T.int64(128), T.int64(128)), "float16"), C_intermediate_1_2: T.Buffer((T.int64(512), T.int64(128)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        # C = AxB
        A_reindex_shared_dyn = T.alloc_buffer((1, 512, 128), "float16", scope="shared.dyn")
        B_reindex_shared_dyn = T.alloc_buffer((1, 128, 512), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_warp = T.alloc_buffer((1, 32, 8, 32, 8), "float16", scope="warp")
        B_reindex_shared_dyn_warp = T.alloc_buffer((1, 8, 32, 32, 8), "float16", scope="warp")
        C_reindex_shared_dyn = T.alloc_buffer((1, 512, 512), "float16", scope="shared.dyn")
        C_reindex_shared_dyn_warp = T.alloc_buffer((1, 32, 32, 32, 8), "float16", scope="warp")
        C_intermediate = T.alloc_buffer((T.int64(512), T.int64(512)), "float16")

        # E = CxD
        A_reindex_shared_dyn_1 = T.alloc_buffer((1, 512, 512), "float16", scope="shared.dyn")
        B_reindex_shared_dyn_1 = T.alloc_buffer((1, 512, 128), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_warp_1 = T.alloc_buffer((1, 32, 32, 32, 8), "float16", scope="warp")
        B_reindex_shared_dyn_warp_1 = T.alloc_buffer((1, 32, 8, 32, 8), "float16", scope="warp")
        C_reindex_shared_dyn_1 = T.alloc_buffer((1, 512, 128), "float16", scope="shared.dyn")
        C_reindex_shared_dyn_warp_1 = T.alloc_buffer((1, 32, 8, 32, 8), "float16", scope="warp")
        C_intermediate_1 = T.alloc_buffer((T.int64(512), T.int64(128)), "float16")
        
        # G = ExF
        A_reindex_shared_dyn_2 = T.alloc_buffer((1, 512, 128), "float16", scope="shared.dyn")
        B_reindex_shared_dyn_2 = T.alloc_buffer((1, 128, 128), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_warp_2 = T.alloc_buffer((1, 32, 8, 32, 8), "float16", scope="warp")
        B_reindex_shared_dyn_warp_2 = T.alloc_buffer((1, 8, 8, 32, 8), "float16", scope="warp")
        C_reindex_shared_dyn_2 = T.alloc_buffer((1, 512, 128), "float16", scope="shared.dyn")
        C_reindex_shared_dyn_warp_2 = T.alloc_buffer((1, 32, 8, 32, 8), "float16", scope="warp")

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


        for ax0 in T.thread_binding(1, thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding(32, thread="blockIdx.y"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(1, thread="blockIdx.x"):
                    for ax1_0_2 in T.thread_binding(1, thread="threadIdx.y"):
                        for ax2_0_2 in T.thread_binding(4, thread="threadIdx.z"):
                            for ax1_0_3_init, ax2_0_3_init in T.grid(1, 2):
                                with T.block("gemm_o_init_2"):
                                    v0_o = T.axis.spatial(1, ax0)
                                    v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused + ax1_0_2 + ax1_0_3_init)
                                    v2_o = T.axis.spatial(8, ax2_0_2 * 2 + ax2_0_3_init)
                                    T.reads()
                                    T.writes(C_reindex_shared_dyn_warp_2[0, v1_o, v2_o, 0:32, 0:8])
                                    with T.block("gemm_init_o_2"):
                                        v1_i_init_o = T.axis.spatial(1, 0)
                                        v2_i_init_o = T.axis.spatial(1, 0)
                                        T.reads()
                                        T.writes(C_reindex_shared_dyn_warp_2[0, v1_o, v2_o, 0:32, 0:8])
                                        C_warp = T.match_buffer(C_reindex_shared_dyn_warp_2[0, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=1)
                                        for tx in T.thread_binding(32, thread="threadIdx.x"):
                                            T.mma_fill("float16", 8, C_warp.data, C_warp.elem_offset)
                            for ax3_0_0 in range(1):
                                for ax0_ax1_ax2_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                        for ax0_ax1_ax2_fused_2 in T.unroll(2, annotations={"pragma_unroll_explicit": 0}):
                                            for ax0_ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax0_ax1_ax2_fused_4 in T.vectorized(8):
                                                    with T.block("A_reindex_shared.dyn_2"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial(512, ax1_0_0_ax2_0_0_fused * 16 + (ax0_ax1_ax2_fused_0 * 2048 + ax0_ax1_ax2_fused_1 * 512 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) // 128)
                                                        v2 = T.axis.spatial(128, (ax0_ax1_ax2_fused_0 * 2048 + ax0_ax1_ax2_fused_1 * 512 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) % 128)
                                                        T.reads(C_intermediate_1[v1, v2])
                                                        T.writes(A_reindex_shared_dyn_2[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "permuted_layout": 0})
                                                        A_reindex_shared_dyn_2[v0, v1, v2] = C_intermediate_1[v1, v2]
                                for ax0_ax1_ax2_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                        for ax0_ax1_ax2_fused_2 in T.unroll(16, annotations={"pragma_unroll_explicit": 0}):
                                            for ax0_ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax0_ax1_ax2_fused_4 in T.vectorized(8):
                                                    with T.block("B_reindex_shared.dyn_2"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial(128, (ax0_ax1_ax2_fused_0 * 16384 + ax0_ax1_ax2_fused_1 * 4096 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) // 128)
                                                        v2 = T.axis.spatial(128, (ax0_ax1_ax2_fused_0 * 16384 + ax0_ax1_ax2_fused_1 * 4096 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) % 128)
                                                        T.reads(F[v1, v2])
                                                        T.writes(B_reindex_shared_dyn_2[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "permuted_layout": 0})
                                                        B_reindex_shared_dyn_2[v0, v1, v2] = F[v1, v2]
                                for ax3_0_1 in range(8):
                                    for ax0_0, ax1_0 in T.grid(1, 1):
                                        with T.block("A_reindex_shared.dyn_warp_o_2"):
                                            v0_o = T.axis.spatial(1, 0)
                                            v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused + ax0_0)
                                            v2_o = T.axis.spatial(8, ax3_0_1 + ax1_0)
                                            T.reads(A_reindex_shared_dyn_2[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(A_reindex_shared_dyn_warp_2[v0_o, v1_o, v2_o, 0:32, 0:8])
                                            T.block_attr({"permuted_layout": 0})
                                            warp = T.match_buffer(A_reindex_shared_dyn_warp_2[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                            shared = T.match_buffer(A_reindex_shared_dyn_2[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                            for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + 8 * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * 16, 1), shared.strides[0] * (tx % 16) + 8 * (tx // 16))
                                    for ax0_0, ax1_0 in T.grid(1, 2):
                                        with T.block("B_reindex_shared.dyn_warp_o_2"):
                                            v0_o = T.axis.spatial(1, 0)
                                            v1_o = T.axis.spatial(8, ax3_0_1 + ax0_0)
                                            v2_o = T.axis.spatial(8, ax2_0_2 * 2 + ax1_0)
                                            T.reads(B_reindex_shared_dyn_2[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(B_reindex_shared_dyn_warp_2[v0_o, v1_o, v2_o, 0:32, 0:8])
                                            T.block_attr({"permuted_layout": 0})
                                            warp = T.match_buffer(B_reindex_shared_dyn_warp_2[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                            shared = T.match_buffer(B_reindex_shared_dyn_2[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                            for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                T.ptx_ldmatrix("float16", T.bool(True), 4, ".b16", warp.data, warp.elem_offset + 8 * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * 16, 1), shared.strides[0] * (tx % 16) + 8 * (tx // 16))
                                    for ax1_0_3, ax2_0_3 in T.grid(1, 2):
                                        with T.block("gemm_o_update_2"):
                                            v0_o = T.axis.spatial(1, ax0)
                                            v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused + ax1_0_2 + ax1_0_3)
                                            v2_o = T.axis.spatial(8, ax2_0_2 * 2 + ax2_0_3)
                                            v3_o = T.axis.reduce(8, ax3_0_0 * 8 + ax3_0_1)
                                            T.reads(C_reindex_shared_dyn_warp_2[0, v1_o, v2_o, 0:32, 0:8], A_reindex_shared_dyn_warp_2[0, v1_o, v3_o, 0:32, 0:8], B_reindex_shared_dyn_warp_2[0, v3_o, v2_o, 0:32, 0:8])
                                            T.writes(C_reindex_shared_dyn_warp_2[0, v1_o, v2_o, 0:32, 0:8])
                                            with T.block("gemm_o_2"):
                                                v1_i_o = T.axis.spatial(1, 0)
                                                v2_i_o = T.axis.spatial(1, 0)
                                                v3_i_o = T.axis.reduce(1, 0)
                                                T.reads(C_reindex_shared_dyn_warp_2[0, v1_o, v2_o, 0:32, 0:8], A_reindex_shared_dyn_warp_2[0, v1_o, v3_o, 0:32, 0:8], B_reindex_shared_dyn_warp_2[0, v3_o, v2_o, 0:32, 0:8])
                                                T.writes(C_reindex_shared_dyn_warp_2[0, v1_o, v2_o, 0:32, 0:8])
                                                A_1 = T.match_buffer(A_reindex_shared_dyn_warp_2[0, v1_o, v3_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                B_1 = T.match_buffer(B_reindex_shared_dyn_warp_2[0, v3_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                C_1 = T.match_buffer(C_reindex_shared_dyn_warp_2[0, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data, A_1.elem_offset + tx * 8, B_1.data, B_1.elem_offset + tx * 8, C_1.data, C_1.elem_offset + tx * 8, T.bool(False))
                                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data, A_1.elem_offset + tx * 8, B_1.data, B_1.elem_offset + tx * 8 + 4, C_1.data, C_1.elem_offset + tx * 8 + 4, T.bool(False))
                            for ax0_0, ax1_0 in T.grid(1, 2):
                                with T.block("C_reindex_shared.dyn_warp_o_2"):
                                    v0_o = T.axis.spatial(1, 0)
                                    v1_o = T.axis.spatial(32, ax1_0_0_ax2_0_0_fused)
                                    v2_o = T.axis.spatial(8, ax2_0_2 * 2 + ax1_0)
                                    T.reads(C_reindex_shared_dyn_warp_2[v0_o, v1_o, v2_o, 0:32, 0:8])
                                    T.writes(C_reindex_shared_dyn_2[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                    C_warp = T.match_buffer(C_reindex_shared_dyn_warp_2[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=1)
                                    C_1 = T.match_buffer(C_reindex_shared_dyn_2[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_1_s0", "C_1_s1"), scope="shared.dyn", offset_factor=1)
                                    for tx in T.thread_binding(32, thread="threadIdx.x"):
                                        T.mma_store("float16", 16, 16, T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_1.strides[0] * 16, 2), C_warp.data, C_warp.elem_offset, C_1.strides[0])
                        for ax0_ax1_ax2_fused_0 in T.unroll(8, annotations={"pragma_unroll_explicit": 0}):
                            for ax0_ax1_ax2_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                for ax0_ax1_ax2_fused_2 in T.vectorized(8):
                                    with T.block("C_reindex_shared.dyn_2"):
                                        v0 = T.axis.spatial(1, 0)
                                        v1 = T.axis.spatial(512, ax1_0_0_ax2_0_0_fused * 16 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2) // 128)
                                        v2 = T.axis.spatial(128, (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2) % 128)
                                        T.reads(C_reindex_shared_dyn_2[v0, v1, v2])
                                        T.writes(C_intermediate_1_2[v1, v2])
                                        C_intermediate_1_2[v1, v2] = C_reindex_shared_dyn_2[v0, v1, v2]

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

    @T.prim_func
    def gemm_2(A: T.Buffer((512, 128), "float16"), B: T.Buffer((128, 128), "float16"), C: T.Buffer((512, 128), "float16")):
        T.func_attr({"dlight.tensorcore_prenormlized": T.bool(True), "op_pattern": 0})
        # with T.block("root"):
        A_reindex_shared_dyn = T.alloc_buffer((1, 512, 128), "float16", scope="shared.dyn")
        B_reindex_shared_dyn = T.alloc_buffer((1, 128, 128), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_warp = T.alloc_buffer((1, 32, 8, 32, 8), "float16", scope="warp")
        B_reindex_shared_dyn_warp = T.alloc_buffer((1, 8, 8, 32, 8), "float16", scope="warp")
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
                                        for ax0_ax1_ax2_fused_2 in T.unroll(16, annotations={"pragma_unroll_explicit": 0}):
                                            for ax0_ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax0_ax1_ax2_fused_4 in T.vectorized(8):
                                                    with T.block("B_reindex_shared.dyn"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial(128, (ax0_ax1_ax2_fused_0 * 16384 + ax0_ax1_ax2_fused_1 * 4096 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) // 128)
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
                                            v2_o = T.axis.spatial(8, ax3_0_1 + ax1_0)
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
                                            v1_o = T.axis.spatial(8, ax3_0_1 + ax0_0)
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
    def main(A: R.Tensor((512, 128), dtype="float16"), B: R.Tensor((128, 512), dtype="float16"), D: R.Tensor((512, 128), dtype="float16"), F: R.Tensor((128, 128), dtype="float16")) -> R.Tensor((512, 128), dtype="float16"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.fused_gemm_0_gemm_1_gemm_2, (A, B, D, F), out_sinfo=R.Tensor((512, 128), dtype="float16"))
            R.output(gv)
        return gv


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
    num_func = 0 #该参数用于检查有几个block

# 建立一个thead_for_map, 后续可以用name 找到1个list，对应的是每个for node
def CollectForPass():
    valid_thread_tags = {
        "threadIdx.x", "threadIdx.y", "threadIdx.z",
        "blockIdx.x", "blockIdx.y", "blockIdx.z"
    }
    
    def _pre_visit(stmt):
        if hasattr(stmt, "thread_binding") and stmt.thread_binding is not None:
            thread_tag = stmt.thread_binding.thread_tag
            if thread_tag in valid_thread_tags:
                ForCollector.thread_for_map[thread_tag].append(stmt)
            if thread_tag == "blockIdx.z":
                ForCollector.num_func += 1
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


# 对primfunc1中 ax2_0_2 for node进行重建，把primfunc2 ax2_0_2 for node加入到primfunc1中
# 将primfunc2 ax2_0_2 改成 ax1_0_2
def ReconstructionPrim():
    thread_for_map = ForCollector.thread_for_map

    def _post_visit(stmt):
        if isinstance(stmt, tir.For) and stmt == thread_for_map["threadIdx.z"][0]:
            extracted_body = [for_node.body.body[0].body for for_node in thread_for_map["blockIdx.x"]]
            new_body = tvm.tir.SeqStmt(extracted_body)
            
            return tvm.tir.For(
                loop_var=stmt.loop_var,
                min=stmt.min,
                extent=stmt.extent,
                kind=stmt.kind,
                body=new_body,
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


# 将primfunc2、3从fused_primfunc中删除
# 删除ax0 for node
def RemoveUselessFunc():
    thread_for_map = defaultdict(list)
    init_A_node = []
    for for_node in thread_for_map["blockIdx.x"][1:]:
        init_A_node.append(for_node.body.body[0].body[1].body[0])

    valid_thread_tags = {
        "threadIdx.x", "threadIdx.y", "threadIdx.z",
        "blockIdx.x", "blockIdx.y", "blockIdx.z"
    }
    
    def _pre_visit(stmt):
        if hasattr(stmt, "thread_binding") and stmt.thread_binding is not None:
            thread_tag = stmt.thread_binding.thread_tag
            if thread_tag in valid_thread_tags:
                thread_for_map[thread_tag].append(stmt)
        return None
    
    def _post_visit(stmt):
        if isinstance(stmt, tvm.tir.For):
            if stmt in thread_for_map["blockIdx.z"][1:]:
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


# 删除fuse以后的函数中对于“A”矩阵的init
# 删除primfunc2 ax3_0_0 for node 的Block(A_reindex_shared.dyn_1)
# (blockIdx.x).body.body[0].body[1].body[0]

# RemoveInit_A的位置应该在ReconstructionPrim这个pass前
def RemoveInit_A():
    thread_for_map = ForCollector.thread_for_map
    
    def _post_visit(stmt):
        target_node = []
        for for_node in thread_for_map["blockIdx.x"][1:]:
            target_node.append(for_node.body.body[0].body[1].body[0])
        if isinstance(stmt, tvm.tir.For) and stmt in target_node:
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


# 替换对应的buffer，可以通过两个map来实现

def ReplaceBufferPass():
    func_to_scope_map = {}
    scope_to_buffer_map = {}
    buffer_replace_map = {}
    num_func = ForCollector.num_func

    import re
    # 后缀提取
    def extract_suffix(name):
        match = re.search(r'_([0-9]+)$', name)
        if match:
            return int(match.group(1))
        return None

    # 前缀提取
    def extract_prefix(name):
        match = re.match(r'^([A-Z])_', name)
        if match:
            return match.group(1)
        return None
        
    def _pre_visit(stmt):
        if isinstance(stmt, tir.For) and stmt.thread_binding == "blockIdx.x":
            C_store_for = stmt.body.body[1]

        if isinstance(stmt, tvm.tir.Block) and stmt.name_hint == "root":
            alloc_buffers_list = stmt.alloc_buffers

            func_buffers = {}
            for i in range(num_func):
                func_buffers[f"func{i}"] = []
            
            for buffer in alloc_buffers_list:
                buffer_name = buffer.name
                suffix = extract_suffix(buffer_name)
                pre_fix = extract_prefix(buffer_name)

                if suffix is not None:
                    func_idx = f"func{suffix}"

                    if suffix < num_func:
                        func_buffers[func_idx].append(buffer)
                else:
                    func_buffers['func0'].append(buffer)
            
            for func_idx, buffers in func_buffers.items():
                if func_idx not in func_to_scope_map:
                    func_to_scope_map[func_idx] = {}
                
                for buffer in buffers:
                    scope = buffer.scope()
                    if scope not in func_to_scope_map[func_idx]:
                        func_to_scope_map[func_idx][scope] = {}
                    
                    buffer_name = buffer.name
                    prefix = extract_prefix(buffer_name)
                    
                    if prefix:
                        buffer_type = prefix
                        func_to_scope_map[func_idx][scope][buffer_type] = buffer

                        scope_to_buffer_map[buffer_name] = {
                            "func": func_idx,
                            "scope": scope, 
                            "type": buffer_type,
                            "buffer": buffer
                        }
            
            # 建立替换关系映射：将func{i+1}的A换成func{i}的C
            for i in range(num_func - 1):
                current_func = f"func{i}"
                next_func = f"func{i+1}"
                
                # if next_func not in buffer_replace_map:
                #     buffer_replace_map[next_func] = {}
                
                valid_scopes = ["shared.dyn", "warp"]
                for scope in func_to_scope_map.get(current_func, {}):
                    if scope not in valid_scopes:
                        continue

                    if scope in func_to_scope_map.get(next_func, {}):
                        C_buffer = func_to_scope_map[current_func][scope]['C']
                        A_buffer = func_to_scope_map[next_func][scope]['A']

                        buffer_replace_map[A_buffer.name] = C_buffer
                        # print(f"replace communication: {next_func}.{scope}.A ({A_buffer.name}) -> {current_func}.{scope}.C ({C_buffer.name})")
            


            # print(f"func_to_scope_map: {func_to_scope_map}")
            # print(f"scope_to_buffer_map: {scope_to_buffer_map}")
        
        return None
    


    def _post_visit(stmt):
        if isinstance(stmt, tir.Block):
            # 处理block的reads区域
            new_reads = []
            for read in stmt.reads:
                if read.buffer.name in buffer_replace_map:
                    new_reads.append(tir.BufferRegion(buffer_replace_map[read.buffer.name], read.region))
                else:
                    new_reads.append(read)
            
            # 处理block的writes区域
            new_writes = []
            for write in stmt.writes:
                if write.buffer.name in buffer_replace_map:
                    new_writes.append(tir.BufferRegion(
                        buffer=buffer_replace_map[write.buffer.name], 
                        region=write.region))
                else:
                    new_writes.append(write)
            
            # 处理block的match_buffers区域
            new_match_buffers = []
            for match in stmt.match_buffers:
                if match.source.buffer.name in buffer_replace_map:
                    new_source = tir.BufferRegion(
                        buffer=buffer_replace_map[match.source.buffer.name],
                        region=match.source.region
                    )
                    new_match_buffers.append(tir.MatchBufferRegion(
                        buffer=match.buffer, 
                        source=new_source
                    ))
                else:
                    new_match_buffers.append(match)
            

            return tir.Block(
                stmt.iter_vars,
                new_reads,
                new_writes,
                stmt.name_hint,
                stmt.body,
                stmt.init,
                stmt.alloc_buffers,
                new_match_buffers,
                stmt.annotations
            )

        elif isinstance(stmt, tir.BufferLoad):
            if stmt.buffer.name in buffer_replace_map:
                new_buffer = buffer_replace_map[stmt.buffer.name]
                return tir.BufferLoad(
                    buffer=new_buffer,
                    indices=stmt.indices,
                    span=stmt.span
                )
        
        elif isinstance(stmt, tir.BufferStore):
            if stmt.buffer.name in buffer_replace_map:
                new_buffer = buffer_replace_map[stmt.buffer.name]
                return tir.BufferStore(
                    buffer=new_buffer,
                    value=stmt.value,
                    indices=stmt.indices,
                    span=stmt.span
                )
            
    def _ftransform(f, mod, ctx):
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(
                f.body,
                _pre_visit,
                _post_visit,
                ["tir.Block", "tir.BufferStore", "tir.BufferLoad"]
            )
        )
    
    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)
                

# 替换所有的轴为primfunc1中的轴
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
def DelGlobalBuffer():
    def _post_visit(stmt):
        if isinstance(stmt, tvm.tir.Block) and stmt.name_hint == "root":
            new_alloc_buffer = [buf for buf in stmt.alloc_buffers if buf.scope != "global"]

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



# 将最终结果存到global memory中
# 该global memory应该在整个primfunc的最后一个函数中
# 需要将最后一个block中的buffer进行替换
# 这部分内容是多余的，可以通过保留最后一个primfunc的最后一个block来将这个删除
def StoreGlobalBuffer():
    target_buffer = None
    thread_for_map = ForCollector.thread_for_map
    C_store_for = None
    last_c_buffer = None

    def _modify_block(stmt):
        nonlocal last_c_buffer
        if isinstance(stmt, tir.Block):
            new_reads = []
            for read in stmt.reads:
                new_reads.append(tir.BufferRegion(
                    buffer=last_c_buffer,
                    region=read.region
                ))

            new_writes = []
            for write in stmt.writes:
                new_writes.append(tir.BufferRegion(
                    buffer=target_buffer,
                    region=write.region
                ))
        
            return tir.Block(
                stmt.iter_vars,
                new_reads,
                new_writes,
                stmt.name_hint,
                stmt.body,
                stmt.init,
                stmt.alloc_buffers,
                stmt.match_buffers,
                stmt.annotations
            )
        
        if isinstance(stmt, tir.BufferLoad):
            new_buffer = last_c_buffer
            return tir.BufferLoad(
                buffer=new_buffer,
                indices=stmt.indices,
                span=stmt.span
            )
        
        if isinstance(stmt, tir.BufferStore):
            new_buffer = target_buffer
            return tir.BufferStore(
                buffer=new_buffer,
                indices=stmt.indices,
                value=stmt.value,
                span=stmt.span
            )
        
        return stmt


    def _pre_visit(stmt):
        nonlocal C_store_for, last_c_buffer
        if isinstance(stmt, tir.For) and hasattr(stmt.thread_binding, "thread_tag") and stmt.thread_binding.thread_tag == "blockIdx.x":
            C_store_for = stmt.body.body[1]
        
        if isinstance(stmt, tir.Block) and stmt.name_hint == "root":
            import re
            c_buffers = {}
            max_suffix = -1

            for buffer in stmt.alloc_buffers:
                match = re.match(r'C_reindex_shared_dyn(?:_(\d+))?$', buffer.name)
                if match:
                    suffix = int(match.group(1)) if match.group(1) else 0
                    c_buffers[suffix] = buffer
                    if suffix > max_suffix:
                        max_suffix = suffix
            
            if max_suffix >= 0:
                last_c_buffer = c_buffers[max_suffix]
        
        return None
                    
        
    # 用最后1个block的for去替换前面的
    def _post_visit(stmt):
        nonlocal C_store_for
        if isinstance(stmt, tir.For) and stmt == C_store_for:
            new_body = tir.stmt_functor.ir_transform(
                stmt.body,
                None, 
                _modify_block,
                ["tir.Block", "tir.BufferLoad", "tir.BufferStore"]
            )

            return tir.For(
                stmt.loop_var,
                stmt.min,
                stmt.extent,
                stmt.kind,
                new_body,
                stmt.thread_binding,
                stmt.annotations
            )

        return stmt

    def _ftransform(func, mod, ctx):
        nonlocal target_buffer

        # 找到input params中的global memory
        for param in func.params:
            buf = func.buffer_map[param]
            if "intermediate" in buf.name:
                target_buffer = buf
                break
        
        return func.with_body(
            tvm.tir.stmt_functor.ir_transform(
                func.body,
                _pre_visit,
                _post_visit,
                ["tir.For", "tir.Block"]
            )
        )
    
    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)




mod = Module
fused_func_name = None
for gv in mod.functions.keys():
    if "fused" in gv.name_hint:
        fused_func_gv = gv
        break

if fused_func_gv: 
    first_func = mod[fused_func_gv]
    new_mod = tvm.IRModule({fused_func_gv: first_func})
    write_mod(new_mod, log_path, "init_mod")
else:
    raise ValueError("No fused function found in the module")

# apply pass
# new_mod = VisitAllNode()(new_mod)
transformed_mod = CollectForPass()(new_mod)
write_mod(transformed_mod, log_path, "CollectForPass")
transformed_mod = ReconstructionPrim()(transformed_mod)
write_mod(transformed_mod, log_path, "ReconstructionPrim")
transformed_mod = RemoveInit_A()(transformed_mod)
write_mod(transformed_mod, log_path, "RemoveInit_A")
transformed_mod = RemoveUselessFunc()(transformed_mod)
write_mod(transformed_mod, log_path, "RemoveUselessFunc")
transformed_mod = SubstituteAxis()(transformed_mod)
write_mod(transformed_mod, log_path, "SubstituteAxis")
transformed_mod = ReplaceBufferPass()(transformed_mod)
write_mod(transformed_mod, log_path, "ReplaceBufferPass")
transformed_mod = StoreGlobalBuffer()(transformed_mod)
write_mod(transformed_mod, log_path, "StoreGlobalBuffer")