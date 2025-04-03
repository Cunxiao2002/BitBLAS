#[version = "0.0.5"]
@fused_gemm_0_gemm_1 = primfn(A_handle: handle, B_handle: handle, D_handle: handle, C_intermediate_1_handle: handle) -> ()
  attr = {"tir.noalias": True}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [512i64, 128i64], []),
             B: Buffer(B_1: Pointer(global float16), float16, [128i64, 512i64], []),
             D: Buffer(D_1: Pointer(global float16), float16, [512i64, 128i64], []),
             C_intermediate_1: Buffer(C_intermediate_1_1: Pointer(global float16), float16, [512i64, 128i64], [])}
  buffer_map = {A_handle: A, B_handle: B, D_handle: D, C_intermediate_1_handle: C_intermediate_1} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_reindex_shared_dyn = alloc_buffer(float16[1, 512, 128])
    B_reindex_shared_dyn = alloc_buffer(float16[1, 128, 512])
    A_reindex_shared_dyn_warp = alloc_buffer(float16[1, 32, 8, 32, 8])
    B_reindex_shared_dyn_warp = alloc_buffer(float16[1, 8, 32, 32, 8])
    C_reindex_shared_dyn = alloc_buffer(float16[1, 512, 512])
    C_reindex_shared_dyn_warp = alloc_buffer(float16[1, 32, 32, 32, 8])
    C_intermediate = alloc_buffer(float16[512i64, 512i64])
    A_reindex_shared_dyn_1 = alloc_buffer(float16[1, 512, 512])
    B_reindex_shared_dyn_1 = alloc_buffer(float16[1, 512, 128])
    A_reindex_shared_dyn_warp_1 = alloc_buffer(float16[1, 32, 32, 32, 8])
    B_reindex_shared_dyn_warp_1 = alloc_buffer(float16[1, 32, 8, 32, 8])
    C_reindex_shared_dyn_1 = alloc_buffer(float16[1, 512, 128])
    C_reindex_shared_dyn_warp_1 = alloc_buffer(float16[1, 32, 8, 32, 8])
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
                    tir.writes([C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8]])
                    block([1, 1], "gemm_init_o") as [v1_i_init_o, v2_i_init_o] {
                      bind(v1_i_init_o, 0)
                      bind(v2_i_init_o, 0)
                      tir.reads([])
                      tir.writes([C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8]])
                      C_warp = match_buffer(C_reindex_shared_dyn_warp[0, v1_o, v2_o, 0:32, 0:8])
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
                            tir.writes([A_reindex_shared_dyn[v0, v1, v2]])
                            tir.attrs({"permuted_layout": 0, "buffer_dim_align": [[0, 1, 16, 8]]})
                            A_reindex_shared_dyn[v0, v1, v2] = A[v1, v2]
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
                            tir.writes([B_reindex_shared_dyn[v0_1, v1_1, v2_1]])
                            tir.attrs({"permuted_layout": 0, "buffer_dim_align": [[0, 1, 16, 8]]})
                            B_reindex_shared_dyn[v0_1, v1_1, v2_1] = B[v1_1, v2_1]
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
                        tir.reads([A_reindex_shared_dyn[v0_o_1, (v1_o_1*16):((v1_o_1*16) + 16), (v2_o_1*16):((v2_o_1*16) + 16)]])
                        tir.writes([A_reindex_shared_dyn_warp[v0_o_1, v1_o_1, v2_o_1, 0:32, 0:8]])
                        tir.attrs({"permuted_layout": 0})
                        warp = match_buffer(A_reindex_shared_dyn_warp[v0_o_1, v1_o_1, v2_o_1, 0:32, 0:8])
                        shared = match_buffer(A_reindex_shared_dyn[v0_o_1, (v1_o_1*16):((v1_o_1*16) + 16), (v2_o_1*16):((v2_o_1*16) + 16)])
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
                        tir.reads([B_reindex_shared_dyn[v0_o_2, (v1_o_2*16):((v1_o_2*16) + 16), (v2_o_2*16):((v2_o_2*16) + 16)]])
                        tir.writes([B_reindex_shared_dyn_warp[v0_o_2, v1_o_2, v2_o_2, 0:32, 0:8]])
                        tir.attrs({"permuted_layout": 0})
                        warp_2 = match_buffer(B_reindex_shared_dyn_warp[v0_o_2, v1_o_2, v2_o_2, 0:32, 0:8])
                        shared_2 = match_buffer(B_reindex_shared_dyn[v0_o_2, (v1_o_2*16):((v1_o_2*16) + 16), (v2_o_2*16):((v2_o_2*16) + 16)])
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
                        tir.reads([C_reindex_shared_dyn_warp[0, v1_o_3, v2_o_3, 0:32, 0:8], A_reindex_shared_dyn_warp[0, v1_o_3, v3_o, 0:32, 0:8], B_reindex_shared_dyn_warp[0, v3_o, v2_o_3, 0:32, 0:8]])
                        tir.writes([C_reindex_shared_dyn_warp[0, v1_o_3, v2_o_3, 0:32, 0:8]])
                        block([1, 1, tir.reduce_axis(0, 1)], "gemm_o") as [v1_i_o, v2_i_o, v3_i_o] {
                          bind(v1_i_o, 0)
                          bind(v2_i_o, 0)
                          bind(v3_i_o, 0)
                          tir.reads([C_reindex_shared_dyn_warp[0, v1_o_3, v2_o_3, 0:32, 0:8], A_reindex_shared_dyn_warp[0, v1_o_3, v3_o, 0:32, 0:8], B_reindex_shared_dyn_warp[0, v3_o, v2_o_3, 0:32, 0:8]])
                          tir.writes([C_reindex_shared_dyn_warp[0, v1_o_3, v2_o_3, 0:32, 0:8]])
                          A_1_1 = match_buffer(A_reindex_shared_dyn_warp[0, v1_o_3, v3_o, 0:32, 0:8])
                          B_1_1 = match_buffer(B_reindex_shared_dyn_warp[0, v3_o, v2_o_3, 0:32, 0:8])
                          C_1 = match_buffer(C_reindex_shared_dyn_warp[0, v1_o_3, v2_o_3, 0:32, 0:8])
                          for (tx_3: int32, 0, 32) "thread_binding" {
                            @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1_2: Pointer(warp float16), (elem_offset_5: int32 + (tx_3*8)), B_1_2: Pointer(warp float16), (elem_offset_6: int32 + (tx_3*8)), C_1_1: Pointer(warp float16), (elem_offset_7: int32 + (tx_3*8)), False, dtype=float16)
                            @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1_2, (elem_offset_5 + (tx_3*8)), B_1_2, ((elem_offset_6 + (tx_3*8)) + 4), C_1_1, ((elem_offset_7 + (tx_3*8)) + 4), False, dtype=float16)
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
                    tir.reads([C_reindex_shared_dyn_warp[v0_o_4, v1_o_4, v2_o_4, 0:32, 0:8]])
                    tir.writes([C_reindex_shared_dyn[v0_o_4, (v1_o_4*16):((v1_o_4*16) + 16), (v2_o_4*16):((v2_o_4*16) + 16)]])
                    C_warp_2 = match_buffer(C_reindex_shared_dyn_warp[v0_o_4, v1_o_4, v2_o_4, 0:32, 0:8])
                    C_1_2 = match_buffer(C_reindex_shared_dyn[v0_o_4, (v1_o_4*16):((v1_o_4*16) + 16), (v2_o_4*16):((v2_o_4*16) + 16)])
                    for (tx_4: int32, 0, 32) "thread_binding" {
                      @tir.mma_store(16, 16, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), C_1_3: Pointer(shared.dyn float16), elem_offset_8: int32, (C_1_s0: int32*16), 2, dtype=handle), C_warp_3: Pointer(warp float16), elem_offset_9: int32, C_1_s0, dtype=float16)
                    }
                }
              }
              for (ax1_0_3_init_1: int32, 0, 1) {
                for (ax2_0_3_init_1: int32, 0, 2) {
                  block([1, 32, 8], "gemm_o_init_1") as [v0_o_5, v1_o_5, v2_o_5] {
                    bind(v0_o_5, ax0_1: int32)
                    bind(v1_o_5, ((ax1_0_0_ax2_0_0_fused_1: int32 + ax1_0_2_1: int32) + ax1_0_3_init_1))
                    bind(v2_o_5, ((ax2_0_2_1: int32*2) + ax2_0_3_init_1))
                    tir.reads([])
                    tir.writes([C_reindex_shared_dyn_warp_1[0, v1_o_5, v2_o_5, 0:32, 0:8]])
                    block([1, 1], "gemm_init_o_1") as [v1_i_init_o_1, v2_i_init_o_1] {
                      bind(v1_i_init_o_1, 0)
                      bind(v2_i_init_o_1, 0)
                      tir.reads([])
                      tir.writes([C_reindex_shared_dyn_warp_1[0, v1_o_5, v2_o_5, 0:32, 0:8]])
                      C_warp_4 = match_buffer(C_reindex_shared_dyn_warp_1[0, v1_o_5, v2_o_5, 0:32, 0:8])
                      for (tx_5: int32, 0, 32) "thread_binding" {
                        @tir.mma_fill(8, C_warp_5: Pointer(warp float16), elem_offset_10: int32, dtype=float16)
                      }
                }
              }
              for (ax3_0_0_1: int32, 0, 4) {
                for (ax0_ax1_ax2_fused_0_2: int32, 0, 1) "thread_binding" {
                  for (ax0_ax1_ax2_fused_1_2: int32, 0, 4) "thread_binding" {
                    for (ax0_ax1_ax2_fused_2_2: int32, 0, 16) "unroll" {
                      for (ax0_ax1_ax2_fused_3_2: int32, 0, 32) "thread_binding" {
                        for (ax0_ax1_ax2_fused_4_2: int32, 0, 8) "vectorized" {
                          block([1, 512, 128], "B_reindex_shared.dyn_1") as [v0_2, v1_2, v2_2] {
                            bind(v0_2, 0)
                            bind(v1_2, ((ax3_0_0_1*128) + floordiv((((((ax0_ax1_ax2_fused_0_2*16384) + (ax0_ax1_ax2_fused_1_2*4096)) + (ax0_ax1_ax2_fused_2_2*256)) + (ax0_ax1_ax2_fused_3_2*8)) + ax0_ax1_ax2_fused_4_2), 128)))
                            bind(v2_2, floormod((((((ax0_ax1_ax2_fused_0_2*16384) + (ax0_ax1_ax2_fused_1_2*4096)) + (ax0_ax1_ax2_fused_2_2*256)) + (ax0_ax1_ax2_fused_3_2*8)) + ax0_ax1_ax2_fused_4_2), 128))
                            tir.reads([D[v1_2, v2_2]])
                            tir.writes([B_reindex_shared_dyn_1[v0_2, v1_2, v2_2]])
                            tir.attrs({"permuted_layout": 0, "buffer_dim_align": [[0, 1, 16, 8]]})
                            B_reindex_shared_dyn_1[v0_2, v1_2, v2_2] = D[v1_2, v2_2]
                        }
                      }
                    }
                  }
                }
                for (ax3_0_1_1: int32, 0, 8) {
                  for (ax0_0_3: int32, 0, 1) {
                    for (ax1_0_5: int32, 0, 1) {
                      block([1, 32, 32], "A_reindex_shared.dyn_warp_o_1") as [v0_o_6, v1_o_6, v2_o_6] {
                        bind(v0_o_6, 0)
                        bind(v1_o_6, (ax1_0_0_ax2_0_0_fused_1 + ax0_0_3))
                        bind(v2_o_6, (((ax3_0_0_1*8) + ax3_0_1_1) + ax1_0_5))
                        tir.reads([C_reindex_shared_dyn[v0_o_6, (v1_o_6*16):((v1_o_6*16) + 16), (v2_o_6*16):((v2_o_6*16) + 16)]])
                        tir.writes([C_reindex_shared_dyn_warp[v0_o_6, v1_o_6, v2_o_6, 0:32, 0:8]])
                        tir.attrs({"permuted_layout": 0})
                        warp_4 = match_buffer(C_reindex_shared_dyn_warp[v0_o_6, v1_o_6, v2_o_6, 0:32, 0:8])
                        shared_4 = match_buffer(C_reindex_shared_dyn[v0_o_6, (v1_o_6*16):((v1_o_6*16) + 16), (v2_o_6*16):((v2_o_6*16) + 16)])
                        for (tx_6: int32, 0, 32) "thread_binding" {
                          @tir.ptx_ldmatrix(False, 4, ".b16", warp_5: Pointer(warp float16), (elem_offset_11: int32 + (8*tx_6)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_5: Pointer(shared.dyn float16), elem_offset_12: int32, (shared_s0_2: int32*16), 1, dtype=handle), ((shared_s0_2*floormod(tx_6, 16)) + (8*floordiv(tx_6, 16))), dtype=float16)
                        }
                    }
                  }
                  for (ax0_0_4: int32, 0, 1) {
                    for (ax1_0_6: int32, 0, 2) {
                      block([1, 32, 8], "B_reindex_shared.dyn_warp_o_1") as [v0_o_7, v1_o_7, v2_o_7] {
                        bind(v0_o_7, 0)
                        bind(v1_o_7, (((ax3_0_0_1*8) + ax3_0_1_1) + ax0_0_4))
                        bind(v2_o_7, ((ax2_0_2_1*2) + ax1_0_6))
                        tir.reads([B_reindex_shared_dyn_1[v0_o_7, (v1_o_7*16):((v1_o_7*16) + 16), (v2_o_7*16):((v2_o_7*16) + 16)]])
                        tir.writes([B_reindex_shared_dyn_warp_1[v0_o_7, v1_o_7, v2_o_7, 0:32, 0:8]])
                        tir.attrs({"permuted_layout": 0})
                        warp_6 = match_buffer(B_reindex_shared_dyn_warp_1[v0_o_7, v1_o_7, v2_o_7, 0:32, 0:8])
                        shared_6 = match_buffer(B_reindex_shared_dyn_1[v0_o_7, (v1_o_7*16):((v1_o_7*16) + 16), (v2_o_7*16):((v2_o_7*16) + 16)])
                        for (tx_7: int32, 0, 32) "thread_binding" {
                          @tir.ptx_ldmatrix(True, 4, ".b16", warp_7: Pointer(warp float16), (elem_offset_13: int32 + (8*tx_7)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_7: Pointer(shared.dyn float16), elem_offset_14: int32, (shared_s0_3: int32*16), 1, dtype=handle), ((shared_s0_3*floormod(tx_7, 16)) + (8*floordiv(tx_7, 16))), dtype=float16)
                        }
                    }
                  }
                  for (ax1_0_3_1: int32, 0, 1) {
                    for (ax2_0_3_1: int32, 0, 2) {
                      block([1, 32, 8, tir.reduce_axis(0, 32)], "gemm_o_update_1") as [v0_o_8, v1_o_8, v2_o_8, v3_o_1] {
                        bind(v0_o_8, ax0_1)
                        bind(v1_o_8, ((ax1_0_0_ax2_0_0_fused_1 + ax1_0_2_1) + ax1_0_3_1))
                        bind(v2_o_8, ((ax2_0_2_1*2) + ax2_0_3_1))
                        bind(v3_o_1, ((ax3_0_0_1*8) + ax3_0_1_1))
                        tir.reads([C_reindex_shared_dyn_warp_1[0, v1_o_8, v2_o_8, 0:32, 0:8], C_reindex_shared_dyn_warp[0, v1_o_8, v3_o_1, 0:32, 0:8], B_reindex_shared_dyn_warp_1[0, v3_o_1, v2_o_8, 0:32, 0:8]])
                        tir.writes([C_reindex_shared_dyn_warp_1[0, v1_o_8, v2_o_8, 0:32, 0:8]])
                        block([1, 1, tir.reduce_axis(0, 1)], "gemm_o_1") as [v1_i_o_1, v2_i_o_1, v3_i_o_1] {
                          bind(v1_i_o_1, 0)
                          bind(v2_i_o_1, 0)
                          bind(v3_i_o_1, 0)
                          tir.reads([C_reindex_shared_dyn_warp_1[0, v1_o_8, v2_o_8, 0:32, 0:8], C_reindex_shared_dyn_warp[0, v1_o_8, v3_o_1, 0:32, 0:8], B_reindex_shared_dyn_warp_1[0, v3_o_1, v2_o_8, 0:32, 0:8]])
                          tir.writes([C_reindex_shared_dyn_warp_1[0, v1_o_8, v2_o_8, 0:32, 0:8]])
                          A_1_3 = match_buffer(C_reindex_shared_dyn_warp[0, v1_o_8, v3_o_1, 0:32, 0:8])
                          B_1_3 = match_buffer(B_reindex_shared_dyn_warp_1[0, v3_o_1, v2_o_8, 0:32, 0:8])
                          C_1_4 = match_buffer(C_reindex_shared_dyn_warp_1[0, v1_o_8, v2_o_8, 0:32, 0:8])
                          for (tx_8: int32, 0, 32) "thread_binding" {
                            @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1_4: Pointer(warp float16), (elem_offset_15: int32 + (tx_8*8)), B_1_4: Pointer(warp float16), (elem_offset_16: int32 + (tx_8*8)), C_1_5: Pointer(warp float16), (elem_offset_17: int32 + (tx_8*8)), False, dtype=float16)
                            @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1_4, (elem_offset_15 + (tx_8*8)), B_1_4, ((elem_offset_16 + (tx_8*8)) + 4), C_1_5, ((elem_offset_17 + (tx_8*8)) + 4), False, dtype=float16)
                          }
                    }
                  }
                }
              }
              for (ax0_0_5: int32, 0, 1) {
                for (ax1_0_7: int32, 0, 2) {
                  block([1, 32, 8], "C_reindex_shared.dyn_warp_o_1") as [v0_o_9, v1_o_9, v2_o_9] {
                    bind(v0_o_9, 0)
                    bind(v1_o_9, ax1_0_0_ax2_0_0_fused_1)
                    bind(v2_o_9, ((ax2_0_2_1*2) + ax1_0_7))
                    tir.reads([C_reindex_shared_dyn_warp_1[v0_o_9, v1_o_9, v2_o_9, 0:32, 0:8]])
                    tir.writes([C_reindex_shared_dyn_1[v0_o_9, (v1_o_9*16):((v1_o_9*16) + 16), (v2_o_9*16):((v2_o_9*16) + 16)]])
                    C_warp_6 = match_buffer(C_reindex_shared_dyn_warp_1[v0_o_9, v1_o_9, v2_o_9, 0:32, 0:8])
                    C_1_6 = match_buffer(C_reindex_shared_dyn_1[v0_o_9, (v1_o_9*16):((v1_o_9*16) + 16), (v2_o_9*16):((v2_o_9*16) + 16)])
                    for (tx_9: int32, 0, 32) "thread_binding" {
                      @tir.mma_store(16, 16, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), C_1_7: Pointer(shared.dyn float16), elem_offset_18: int32, (C_1_s0_1: int32*16), 2, dtype=handle), C_warp_7: Pointer(warp float16), elem_offset_19: int32, C_1_s0_1, dtype=float16)
                    }
                }
              }
            }
            for (ax0_ax1_ax2_fused_0_3: int32, 0, 32) "unroll" {
              for (ax0_ax1_ax2_fused_1_3: int32, 0, 32) "thread_binding" {
                for (ax0_ax1_ax2_fused_2_3: int32, 0, 8) "vectorized" {
                  block([1, 512, 512], "C_reindex_shared.dyn") as [v0_3, v1_3, v2_3] {
                    bind(v0_3, 0)
                    bind(v1_3, ((ax1_0_0_ax2_0_0_fused*16) + floordiv((((ax0_ax1_ax2_fused_0_3*256) + (ax0_ax1_ax2_fused_1_3*8)) + ax0_ax1_ax2_fused_2_3), 512)))
                    bind(v2_3, floormod((((ax0_ax1_ax2_fused_0_3*256) + (ax0_ax1_ax2_fused_1_3*8)) + ax0_ax1_ax2_fused_2_3), 512))
                    tir.reads([C_reindex_shared_dyn_1[v0_3, v1_3, v2_3]])
                    tir.writes([C_intermediate[v1_3, v2_3]])
                    C_intermediate[v1_3, v2_3] = C_reindex_shared_dyn[v0_3, v1_3, v2_3]
                }
              }
            }
          }
        }
      }
    }
}



/* For debugging purposes the metadata section has been omitted.
 * If you would like to see the full metadata section you can set the 
 * option to `True` when invoking `astext`. 
 */