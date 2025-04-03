from ast import For, stmt
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
    @T.prim_func
    def fused_dense_relu_0(input0: T.Buffer((T.int64(2073600), T.int64(64)), "float16"), param_0: T.Buffer((T.int64(64), T.int64(64)), "float16"), T_relu_intermediate: T.Buffer((T.int64(2073600), T.int64(64)), "float16")):
        T.func_attr({"dlight.tensorcore_prenormlized": T.bool(True), "op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        input0_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(2073600), T.int64(64)), "float16", scope="shared.dyn")
        param_0_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(64)), "float16", scope="shared.dyn")
        input0_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), T.int64(129600), T.int64(4), T.int64(32), T.int64(8)), "float16", scope="warp")
        param_0_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), T.int64(4), T.int64(4), T.int64(32), T.int64(8)), "float16", scope="warp")
        T_matmul_NT_intermediate_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(2073600), T.int64(64)), "float16", scope="shared.dyn")
        T_matmul_NT_intermediate_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), T.int64(129600), T.int64(4), T.int64(32), T.int64(8)), "float16", scope="warp")
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
                                        T.writes(T_relu_intermediate[v1, v2])
                                        T_relu_intermediate[v1, v2] = T.max(T_matmul_NT_intermediate_reindex_shared_dyn[v0, v1, v2], T.float16(0))

    @T.prim_func
    def fused_dense_relu_1(input0: T.Buffer((T.int64(2073600), T.int64(64)), "float16"), param_0: T.Buffer((T.int64(64), T.int64(64)), "float16"), T_relu_intermediate: T.Buffer((T.int64(2073600), T.int64(64)), "float16")):
        T.func_attr({"dlight.tensorcore_prenormlized": T.bool(True), "op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        input0_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(2073600), T.int64(64)), "float16", scope="shared.dyn")
        param_0_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(64)), "float16", scope="shared.dyn")
        input0_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), T.int64(129600), T.int64(4), T.int64(32), T.int64(8)), "float16", scope="warp")
        param_0_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), T.int64(4), T.int64(4), T.int64(32), T.int64(8)), "float16", scope="warp")
        T_matmul_NT_intermediate_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(2073600), T.int64(64)), "float16", scope="shared.dyn")
        T_matmul_NT_intermediate_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), T.int64(129600), T.int64(4), T.int64(32), T.int64(8)), "float16", scope="warp")
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
                                        T.writes(T_relu_intermediate[v1, v2])
                                        T_relu_intermediate[v1, v2] = T.max(T_matmul_NT_intermediate_reindex_shared_dyn[v0, v1, v2], T.float16(0))

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
        for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding(T.int64(21600), thread="blockIdx.y"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                    for ax1_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax2_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                            for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(3), T.int64(2)):
                                with T.block("T_matmul_NT_o_init_1"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0)
                                    v1_o = T.axis.spatial(T.int64(129600), ax1_0_0_ax2_0_0_fused * T.int64(6) + ax1_0_2 * T.int64(3) + ax1_0_3_init)
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
                                        for ax0_ax1_ax2_fused_2 in T.unroll(T.int64(3), annotations={"pragma_unroll_explicit": 0}):
                                            for ax0_ax1_ax2_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                                for ax0_ax1_ax2_fused_4 in T.vectorized(T.int64(8)):
                                                    with T.block("input0_reindex_shared.dyn_1"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(2073600), ax1_0_0_ax2_0_0_fused * T.int64(96) + (ax0_ax1_ax2_fused_0 * T.int64(1536) + ax0_ax1_ax2_fused_1 * T.int64(768) + ax0_ax1_ax2_fused_2 * T.int64(256) + ax0_ax1_ax2_fused_3 * T.int64(8) + ax0_ax1_ax2_fused_4) // T.int64(32))
                                                        v2 = T.axis.spatial(T.int64(64), ax3_0_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(1536) + ax0_ax1_ax2_fused_1 * T.int64(768) + ax0_ax1_ax2_fused_2 * T.int64(256) + ax0_ax1_ax2_fused_3 * T.int64(8) + ax0_ax1_ax2_fused_4) % T.int64(32))
                                                        T.reads(T_relu_intermediate_intermediate[v1, v2])
                                                        T.writes(input0_reindex_shared_dyn_1[v0, v1, v2])
                                                        T.block_attr({"permuted_layout": 1})
                                                        input0_reindex_shared_dyn_1[v0, v1, v2] = T_relu_intermediate_intermediate[v1, v2]
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
                                            v1_o = T.axis.spatial(T.int64(129600), ax1_0_0_ax2_0_0_fused * T.int64(6) + ax1_0_2 * T.int64(3) + ax0_0)
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
                                            v0_o = T.axis.spatial(T.int64(1), ax0)
                                            v1_o = T.axis.spatial(T.int64(129600), ax1_0_0_ax2_0_0_fused * T.int64(6) + ax1_0_2 * T.int64(3) + ax1_0_3)
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
                                    v1_o = T.axis.spatial(T.int64(129600), ax1_0_0_ax2_0_0_fused * T.int64(6) + ax1_0_2 * T.int64(3) + ax0_0)
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
                                        v1 = T.axis.spatial(T.int64(2073600), ax1_0_0_ax2_0_0_fused * T.int64(96) + ax1_0_2 * T.int64(48) + (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2) // T.int64(64))
                                        v2 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2) % T.int64(64))
                                        T.reads(T_matmul_NT_intermediate_reindex_shared_dyn_1[v0, v1, v2])
                                        T.writes(T_relu_intermediate_intermediate_1[v1, v2])
                                        T_relu_intermediate_intermediate_1[v1, v2] = T.max(T_matmul_NT_intermediate_reindex_shared_dyn_1[v0, v1, v2], T.float16(0))

    @R.function
    def main(input0: R.Tensor((2073600, 64), dtype="float16"), param_0: R.Tensor((64, 64), dtype="float16"), param_1: R.Tensor((64, 64), dtype="float16")) -> R.Tensor((2073600, 64), dtype="float16"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.fused_fused_dense_relu_0_fused_dense_relu_1, (input0, param_0, param_1), out_sinfo=R.Tensor((2073600, 64), dtype="float16"))
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

def ReconstructionPrim():
    thread_for_map = ForCollector.thread_for_map

    def _post_visit(stmt):
        if isinstance(stmt, tir.For) and stmt == thread_for_map["threadIdx.y"][0]:
            extracted_body = [for_node.body.body for for_node in thread_for_map["blockIdx.x"]]
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
        # match = re.match(r'^([A-Z])_', name)
        # if match:
        #     return match.group(1)
        if name.startswith('input'):
            return 'input'
        if name.startswith('param'):
            return 'param'
        if name.startswith('T_matmul'):
            return 'T_matmul'
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
                        C_buffer = func_to_scope_map[current_func][scope]['T_matmul']
                        A_buffer = func_to_scope_map[next_func][scope]['input']

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
    del thread_for_map["threadIdx.z"]
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
# 已废除
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
                match = re.match(r'T_matmul_NT_intermediate_reindex_shared_dyn(?:_(\d+))?$', buffer.name)
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



# 替换中间relu的buffer
# 将block中的 writes buffer 换成 reads buffer
def ReluBufferReplace():
    blockIdx_for = None
    load_indices = None
    reads_bufferRegion = None

    def _pre_modify_block(stmt):
        nonlocal reads_bufferRegion
        if isinstance(stmt, tir.Block):
            for read in stmt.reads:
                reads_bufferRegion = read


    def _modify_block(stmt):
        nonlocal load_indices, reads_bufferRegion
        new_writes = []
        new_writes.append(reads_bufferRegion)
        if isinstance(stmt, tir.Block):
            return tir.Block(
                stmt.iter_vars,
                stmt.reads,
                new_writes,
                stmt.name_hint,
                stmt.body,
                stmt.init,
                stmt.alloc_buffers,
                stmt.match_buffers,
                stmt.annotations
            )
        if isinstance(stmt, tir.BufferLoad):
            load_indices = stmt.indices
        
        if isinstance(stmt, tir.BufferStore):
            return tir.BufferStore(
                buffer=reads_bufferRegion.buffer,
                indices=load_indices,
                value=stmt.value
            )
    
    def _pre_visit(stmt):
        nonlocal blockIdx_for
        if isinstance(stmt, tir.For) and stmt.thread_binding is not None:
            if tvm.tir.all(stmt.thread_binding.thread_tag == "blockIdx.x"):
                blockIdx_for = stmt


    def _post_visit(stmt):
        nonlocal blockIdx_for
        actual_index = len(blockIdx_for.body.body) - 3
        if blockIdx_for and stmt == blockIdx_for.body.body[actual_index]:
            new_body = tir.stmt_functor.ir_transform(
                stmt.body,
                _pre_modify_block,
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
    
    def _ftransform(func, mod, ctx):
        return func.with_body(
            tvm.tir.stmt_functor.ir_transform(
                func.body,
                _pre_visit,
                _post_visit,
                ["tir.For", "tir.Block"]
            )
        )
    
    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)


# trick to solve swizzle problem
def SolveSwizzleProblem():
    blockIdx_for = None
    
    def _pre_visit(stmt):
        nonlocal blockIdx_for
        if isinstance(stmt, tir.For) and stmt.thread_binding is not None:
            if tvm.tir.all(stmt.thread_binding.thread_tag == "blockIdx.x"):
                blockIdx_for = stmt

    def _post_visit(stmt):
        nonlocal blockIdx_for
        # stmt.body.body[-2].body[1].body[1].body[0].body.body 是
        # block(input0_reindex_shared.dyn_warp_o_1)
        target_block = blockIdx_for.body.body[-2].body[1].body[1].body[0].body.body.block
        if isinstance(stmt, tir.Block) and stmt == target_block:
            # 去掉annotations的注释，在这一步不使用swizzle
            return tir.Block(
                stmt.iter_vars,
                stmt.reads,
                stmt.writes,
                stmt.name_hint,
                stmt.body,
                stmt.init,
                stmt.alloc_buffers,
                stmt.match_buffers,
            )

    def _ftransform(f, mod, ctx):
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(
                f.body,
                _pre_visit,
                _post_visit,
                ["tir.Block", "tir.For"]
            )
        )
    
    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)

mod = Module
fused_func_name = None
for gv in mod.functions.keys():
    if gv.name_hint == "fused_fused_dense_relu_0_fused_dense_relu_1":
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
transformed_mod = ReluBufferReplace()(transformed_mod)
write_mod(transformed_mod, log_path, "ReluBufferReplace")
transformed_mod = SolveSwizzleProblem()(transformed_mod)
write_mod(transformed_mod, log_path, "SolveSwizzleProblem")