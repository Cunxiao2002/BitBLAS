#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"global_symbol": "gemm", "dlight.tensorcore_prenormlized": True}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [512, 128], []),
             B: Buffer(B_1: Pointer(global float16), float16, [128, 512], []),
             C: Buffer(C_1: Pointer(global float16), float16, [512, 512], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_reindex_shared.dyn = alloc_buffer(float16[1, 512, 128])
    B_reindex_shared.dyn = alloc_buffer(float16[1, 128, 512])
    A_reindex_shared.dyn_warp = alloc_buffer(float16[1, 32, 8, 32, 8])
    B_reindex_shared.dyn_warp = alloc_buffer(float16[1, 8, 32, 32, 8])
    C_reindex_shared.dyn = alloc_buffer(float16[1, 512, 512])
    C_reindex_shared.dyn_warp = alloc_buffer(float16[1, 32, 32, 32, 8])
    for (ax0: int32, 0, 1) "thread_binding" {
      for (ax1_0_0_ax2_0_0_fused: int32, 0, 32) "thread_binding" {
        for (ax1_0_1_ax2_0_1_fused: int32, 0, 1) "thread_binding" {
          for (ax1_0_2: int32, 0, 1) "thread_binding" {
            for (ax2_0_2: int32, 0, 4) "thread_binding" {
              for (ax1_0_3_init: int32, 0, 1) {
                for (ax2_0_3_init: int32, 0, 8) {
                  block([1, 32, 32], "gemm_o_init") as [v0_o, v1_o, v2_o] {
                    bind(v0_o, ax0)
                    bind(v1_o, ((ax1_0_0_ax2_0_0_fused + ax1_0_2) + ax1_0_3_init))
                    bind(v2_o, ((ax2_0_2*8) + ax2_0_3_init))
                    tir.reads([])
                    tir.writes([C_reindex_shared.dyn_warp[0, v1_o, v2_o, 0:32, 0:8]])
                    block([1, 1], "gemm_init_o") as [v1_i_init_o, v2_i_init_o] {
                      bind(v1_i_init_o, 0)
                      bind(v2_i_init_o, 0)
                      tir.reads([])
                      tir.writes([C_reindex_shared.dyn_warp[0, v1_o, v2_o, 0:32, 0:8]])
                      C_warp = match_buffer(C_reindex_shared.dyn_warp[0, v1_o, v2_o, 0:32, 0:8])
                      for (tx: int32, 0, 32) "thread_binding" {
                        @tir.mma_fill(8, C_warp_1: Pointer(warp float16), elem_offset: int32, dtype=float16)
                      }
                }
              }
              for (ax3_0_0: int32, 0, 1) {
                for (ax0_ax1_ax2_fused_0: int32, 0, 1) "thread_binding" {
                  for (ax0_ax1_ax2_fused_1: int32, 0, 4) "thread_binding" {
                    for (ax0_ax1_ax2_fused_2: int32, 0, 2) "unroll" {
                      for (ax0_ax1_ax2_fused_3: int32, 0, 32) "thread_binding" {
                        for (ax0_ax1_ax2_fused_4: int32, 0, 8) "vectorized" {
                          block([1, 512, 128], "A_reindex_shared.dyn") as [v0, v1, v2] {
                            bind(v0, 0)
                            bind(v1, ((ax1_0_0_ax2_0_0_fused*16) + floordiv((((((ax0_ax1_ax2_fused_0*2048) + (ax0_ax1_ax2_fused_1*512)) + (ax0_ax1_ax2_fused_2*256)) + (ax0_ax1_ax2_fused_3*8)) + ax0_ax1_ax2_fused_4), 128)))
                            bind(v2, floormod((((((ax0_ax1_ax2_fused_0*2048) + (ax0_ax1_ax2_fused_1*512)) + (ax0_ax1_ax2_fused_2*256)) + (ax0_ax1_ax2_fused_3*8)) + ax0_ax1_ax2_fused_4), 128))
                            tir.reads([A[v1, v2]])
                            tir.writes([A_reindex_shared.dyn[v0, v1, v2]])
                            tir.attrs({"permuted_layout": 0, "buffer_dim_align": [[0, 1, 16, 8]]})
                            A_reindex_shared.dyn[v0, v1, v2] = A[v1, v2]
                        }
                      }
                    }
                  }
                }
                for (ax0_ax1_ax2_fused_0_1: int32, 0, 1) "thread_binding" {
                  for (ax0_ax1_ax2_fused_1_1: int32, 0, 4) "thread_binding" {
                    for (ax0_ax1_ax2_fused_2_1: int32, 0, 64) "unroll" {
                      for (ax0_ax1_ax2_fused_3_1: int32, 0, 32) "thread_binding" {
                        for (ax0_ax1_ax2_fused_4_1: int32, 0, 8) "vectorized" {
                          block([1, 128, 512], "B_reindex_shared.dyn") as [v0_1, v1_1, v2_1] {
                            bind(v0_1, 0)
                            bind(v1_1, floordiv((((((ax0_ax1_ax2_fused_0_1*65536) + (ax0_ax1_ax2_fused_1_1*16384)) + (ax0_ax1_ax2_fused_2_1*256)) + (ax0_ax1_ax2_fused_3_1*8)) + ax0_ax1_ax2_fused_4_1), 512))
                            bind(v2_1, floormod((((((ax0_ax1_ax2_fused_0_1*65536) + (ax0_ax1_ax2_fused_1_1*16384)) + (ax0_ax1_ax2_fused_2_1*256)) + (ax0_ax1_ax2_fused_3_1*8)) + ax0_ax1_ax2_fused_4_1), 512))
                            tir.reads([B[v1_1, v2_1]])
                            tir.writes([B_reindex_shared.dyn[v0_1, v1_1, v2_1]])
                            tir.attrs({"permuted_layout": 0, "buffer_dim_align": [[0, 1, 16, 8]]})
                            B_reindex_shared.dyn[v0_1, v1_1, v2_1] = B[v1_1, v2_1]
                        }
                      }
                    }
                  }
                }
                for (ax3_0_1: int32, 0, 8) {
                  for (ax0_0: int32, 0, 1) {
                    for (ax1_0: int32, 0, 1) {
                      block([1, 32, 8], "A_reindex_shared.dyn_warp_o") as [v0_o_1, v1_o_1, v2_o_1] {
                        bind(v0_o_1, 0)
                        bind(v1_o_1, (ax1_0_0_ax2_0_0_fused + ax0_0))
                        bind(v2_o_1, (ax3_0_1 + ax1_0))
                        tir.reads([A_reindex_shared.dyn[v0_o_1, (v1_o_1*16):((v1_o_1*16) + 16), (v2_o_1*16):((v2_o_1*16) + 16)]])
                        tir.writes([A_reindex_shared.dyn_warp[v0_o_1, v1_o_1, v2_o_1, 0:32, 0:8]])
                        tir.attrs({"permuted_layout": 0})
                        warp = match_buffer(A_reindex_shared.dyn_warp[v0_o_1, v1_o_1, v2_o_1, 0:32, 0:8])
                        shared = match_buffer(A_reindex_shared.dyn[v0_o_1, (v1_o_1*16):((v1_o_1*16) + 16), (v2_o_1*16):((v2_o_1*16) + 16)])
                        for (tx_1: int32, 0, 32) "thread_binding" {
                          @tir.ptx_ldmatrix(False, 4, ".b16", warp_1: Pointer(warp float16), (elem_offset_1: int32 + (8*tx_1)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_1: Pointer(shared.dyn float16), elem_offset_2: int32, (shared_s0: int32*16), 1, dtype=handle), ((shared_s0*floormod(tx_1, 16)) + (8*floordiv(tx_1, 16))), dtype=float16)
                        }
                    }
                  }
                  for (ax0_0_1: int32, 0, 1) {
                    for (ax1_0_1: int32, 0, 8) {
                      block([1, 8, 32], "B_reindex_shared.dyn_warp_o") as [v0_o_2, v1_o_2, v2_o_2] {
                        bind(v0_o_2, 0)
                        bind(v1_o_2, (ax3_0_1 + ax0_0_1))
                        bind(v2_o_2, ((ax2_0_2*8) + ax1_0_1))
                        tir.reads([B_reindex_shared.dyn[v0_o_2, (v1_o_2*16):((v1_o_2*16) + 16), (v2_o_2*16):((v2_o_2*16) + 16)]])
                        tir.writes([B_reindex_shared.dyn_warp[v0_o_2, v1_o_2, v2_o_2, 0:32, 0:8]])
                        tir.attrs({"permuted_layout": 0})
                        warp_2 = match_buffer(B_reindex_shared.dyn_warp[v0_o_2, v1_o_2, v2_o_2, 0:32, 0:8])
                        shared_2 = match_buffer(B_reindex_shared.dyn[v0_o_2, (v1_o_2*16):((v1_o_2*16) + 16), (v2_o_2*16):((v2_o_2*16) + 16)])
                        for (tx_2: int32, 0, 32) "thread_binding" {
                          @tir.ptx_ldmatrix(True, 4, ".b16", warp_3: Pointer(warp float16), (elem_offset_3: int32 + (8*tx_2)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_3: Pointer(shared.dyn float16), elem_offset_4: int32, (shared_s0_1: int32*16), 1, dtype=handle), ((shared_s0_1*floormod(tx_2, 16)) + (8*floordiv(tx_2, 16))), dtype=float16)
                        }
                    }
                  }
                  for (ax1_0_3: int32, 0, 1) {
                    for (ax2_0_3: int32, 0, 8) {
                      block([1, 32, 32, tir.reduce_axis(0, 8)], "gemm_o_update") as [v0_o_3, v1_o_3, v2_o_3, v3_o] {
                        bind(v0_o_3, ax0)
                        bind(v1_o_3, ((ax1_0_0_ax2_0_0_fused + ax1_0_2) + ax1_0_3))
                        bind(v2_o_3, ((ax2_0_2*8) + ax2_0_3))
                        bind(v3_o, ((ax3_0_0*8) + ax3_0_1))
                        tir.reads([C_reindex_shared.dyn_warp[0, v1_o_3, v2_o_3, 0:32, 0:8], A_reindex_shared.dyn_warp[0, v1_o_3, v3_o, 0:32, 0:8], B_reindex_shared.dyn_warp[0, v3_o, v2_o_3, 0:32, 0:8]])
                        tir.writes([C_reindex_shared.dyn_warp[0, v1_o_3, v2_o_3, 0:32, 0:8]])
                        block([1, 1, tir.reduce_axis(0, 1)], "gemm_o") as [v1_i_o, v2_i_o, v3_i_o] {
                          bind(v1_i_o, 0)
                          bind(v2_i_o, 0)
                          bind(v3_i_o, 0)
                          tir.reads([C_reindex_shared.dyn_warp[0, v1_o_3, v2_o_3, 0:32, 0:8], A_reindex_shared.dyn_warp[0, v1_o_3, v3_o, 0:32, 0:8], B_reindex_shared.dyn_warp[0, v3_o, v2_o_3, 0:32, 0:8]])
                          tir.writes([C_reindex_shared.dyn_warp[0, v1_o_3, v2_o_3, 0:32, 0:8]])
                          A_2 = match_buffer(A_reindex_shared.dyn_warp[0, v1_o_3, v3_o, 0:32, 0:8])
                          B_2 = match_buffer(B_reindex_shared.dyn_warp[0, v3_o, v2_o_3, 0:32, 0:8])
                          C_2 = match_buffer(C_reindex_shared.dyn_warp[0, v1_o_3, v2_o_3, 0:32, 0:8])
                          for (tx_3: int32, 0, 32) "thread_binding" {
                            @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_3: Pointer(warp float16), (elem_offset_5: int32 + (tx_3*8)), B_3: Pointer(warp float16), (elem_offset_6: int32 + (tx_3*8)), C_3: Pointer(warp float16), (elem_offset_7: int32 + (tx_3*8)), False, dtype=float16)
                            @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_3, (elem_offset_5 + (tx_3*8)), B_3, ((elem_offset_6 + (tx_3*8)) + 4), C_3, ((elem_offset_7 + (tx_3*8)) + 4), False, dtype=float16)
                          }
                    }
                  }
                }
              }
              for (ax0_0_2: int32, 0, 1) {
                for (ax1_0_4: int32, 0, 8) {
                  block([1, 32, 32], "C_reindex_shared.dyn_warp_o") as [v0_o_4, v1_o_4, v2_o_4] {
                    bind(v0_o_4, 0)
                    bind(v1_o_4, ax1_0_0_ax2_0_0_fused)
                    bind(v2_o_4, ((ax2_0_2*8) + ax1_0_4))
                    tir.reads([C_reindex_shared.dyn_warp[v0_o_4, v1_o_4, v2_o_4, 0:32, 0:8]])
                    tir.writes([C_reindex_shared.dyn[v0_o_4, (v1_o_4*16):((v1_o_4*16) + 16), (v2_o_4*16):((v2_o_4*16) + 16)]])
                    C_warp_2 = match_buffer(C_reindex_shared.dyn_warp[v0_o_4, v1_o_4, v2_o_4, 0:32, 0:8])
                    C_4 = match_buffer(C_reindex_shared.dyn[v0_o_4, (v1_o_4*16):((v1_o_4*16) + 16), (v2_o_4*16):((v2_o_4*16) + 16)])
                    for (tx_4: int32, 0, 32) "thread_binding" {
                      @tir.mma_store(16, 16, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), C_5: Pointer(shared.dyn float16), elem_offset_8: int32, (C_s0: int32*16), 2, dtype=handle), C_warp_3: Pointer(warp float16), elem_offset_9: int32, C_s0, dtype=float16)
                    }
                }
              }
            }
            for (ax0_ax1_ax2_fused_0_2: int32, 0, 32) "unroll" {
              for (ax0_ax1_ax2_fused_1_2: int32, 0, 32) "thread_binding" {
                for (ax0_ax1_ax2_fused_2_2: int32, 0, 8) "vectorized" {
                  block([1, 512, 512], "C_reindex_shared.dyn") as [v0_2, v1_2, v2_2] {
                    bind(v0_2, 0)
                    bind(v1_2, ((ax1_0_0_ax2_0_0_fused*16) + floordiv((((ax0_ax1_ax2_fused_0_2*256) + (ax0_ax1_ax2_fused_1_2*8)) + ax0_ax1_ax2_fused_2_2), 512)))
                    bind(v2_2, floormod((((ax0_ax1_ax2_fused_0_2*256) + (ax0_ax1_ax2_fused_1_2*8)) + ax0_ax1_ax2_fused_2_2), 512))
                    tir.reads([C_reindex_shared.dyn[v0_2, v1_2, v2_2]])
                    tir.writes([C[v1_2, v2_2]])
                    C[v1_2, v2_2] = C_reindex_shared.dyn[v0_2, v1_2, v2_2]
                }
              }
            }
          }
        }
      }
    }
}

#[metadata]
{
  "root": 1, 
  "nodes": [
    {
      "type_key": ""
    }, 
    {
      "type_key": "Map", 
      "keys": [
        "IntImm"
      ], 
      "data": [2]
    }, 
    {
      "type_key": "Array", 
      "data": [3]
    }, 
    {
      "type_key": "IntImm", 
      "attrs": {
        "dtype": "bool", 
        "span": "0", 
        "value": "1"
      }
    }
  ], 
  "b64ndarrays": [], 
  "attrs": {"tvm_version": "0.17.dev0"}
}