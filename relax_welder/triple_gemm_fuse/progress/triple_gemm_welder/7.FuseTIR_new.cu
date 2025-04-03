#[version = "0.0.5"]
@fused_gemm_0_gemm_1_gemm_2 = primfn(p_A: handle, p_B: handle, p_D: handle, p_F: handle, p_output0: handle) -> ()
  attr = {"tir.noalias": True}
  buffers = {A: Buffer(A_1: Pointer(float16), float16, [512i64, 128i64], []),
             B: Buffer(B_1: Pointer(float16), float16, [128i64, 512i64], []),
             D: Buffer(D_1: Pointer(float16), float16, [512i64, 128i64], []),
             F: Buffer(F_1: Pointer(float16), float16, [128i64, 128i64], []),
             C_intermediate_1_2: Buffer(C: Pointer(global float16), float16, [512i64, 128i64], [])}
  buffer_map = {p_A: A, p_B: B, p_D: D, p_F: F, p_output0: C_intermediate_1_2} {
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
    C_intermediate_1 = alloc_buffer(float16[512i64, 128i64])
    A_reindex_shared_dyn_2 = alloc_buffer(float16[1, 512, 128])
    B_reindex_shared_dyn_2 = alloc_buffer(float16[1, 128, 128])
    A_reindex_shared_dyn_warp_2 = alloc_buffer(float16[1, 32, 8, 32, 8])
    B_reindex_shared_dyn_warp_2 = alloc_buffer(float16[1, 8, 8, 32, 8])
    C_reindex_shared_dyn_2 = alloc_buffer(float16[1, 512, 128])
    C_reindex_shared_dyn_warp_2 = alloc_buffer(float16[1, 32, 8, 32, 8])
     {
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
              }
              for (ax0_ax1_ax2_fused_0_2: int32, 0, 32) "unroll" {
                for (ax0_ax1_ax2_fused_1_2: int32, 0, 32) "thread_binding" {
                  for (ax0_ax1_ax2_fused_2_2: int32, 0, 8) "vectorized" {
                    block([1, 512, 512], "C_reindex_shared.dyn") as [v0_2, v1_2, v2_2] {
                      bind(v0_2, 0)
                      bind(v1_2, ((ax1_0_0_ax2_0_0_fused*16) + floordiv((((ax0_ax1_ax2_fused_0_2*256) + (ax0_ax1_ax2_fused_1_2*8)) + ax0_ax1_ax2_fused_2_2), 512)))
                      bind(v2_2, floormod((((ax0_ax1_ax2_fused_0_2*256) + (ax0_ax1_ax2_fused_1_2*8)) + ax0_ax1_ax2_fused_2_2), 512))
                      tir.reads([C_reindex_shared_dyn[v0_2, v1_2, v2_2]])
                      tir.writes([C_intermediate[v1_2, v2_2]])
                      C_intermediate[v1_2, v2_2] = C_reindex_shared_dyn[v0_2, v1_2, v2_2]
                  }
                }
              }
            }
          }
        }
      }
      for (ax0_1: int32, 0, 1) "thread_binding" {
        for (ax1_0_0_ax2_0_0_fused_1: int32, 0, 32) "thread_binding" {
          for (ax1_0_1_ax2_0_1_fused_1: int32, 0, 1) "thread_binding" {
            for (ax1_0_2_1: int32, 0, 1) "thread_binding" {
              for (ax2_0_2_1: int32, 0, 4) "thread_binding" {
                for (ax1_0_3_init_1: int32, 0, 1) {
                  for (ax2_0_3_init_1: int32, 0, 2) {
                    block([1, 32, 8], "gemm_o_init_1") as [v0_o_5, v1_o_5, v2_o_5] {
                      bind(v0_o_5, ax0_1)
                      bind(v1_o_5, ((ax1_0_0_ax2_0_0_fused_1 + ax1_0_2_1) + ax1_0_3_init_1))
                      bind(v2_o_5, ((ax2_0_2_1*2) + ax2_0_3_init_1))
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
                  for (ax0_ax1_ax2_fused_0_3: int32, 0, 1) "thread_binding" {
                    for (ax0_ax1_ax2_fused_1_3: int32, 0, 4) "thread_binding" {
                      for (ax0_ax1_ax2_fused_2_3: int32, 0, 2) "unroll" {
                        for (ax0_ax1_ax2_fused_3_2: int32, 0, 32) "thread_binding" {
                          for (ax0_ax1_ax2_fused_4_2: int32, 0, 8) "vectorized" {
                            block([1, 512, 512], "A_reindex_shared.dyn_1") as [v0_3, v1_3, v2_3] {
                              bind(v0_3, 0)
                              bind(v1_3, ((ax1_0_0_ax2_0_0_fused_1*16) + floordiv((((((ax0_ax1_ax2_fused_0_3*2048) + (ax0_ax1_ax2_fused_1_3*512)) + (ax0_ax1_ax2_fused_2_3*256)) + (ax0_ax1_ax2_fused_3_2*8)) + ax0_ax1_ax2_fused_4_2), 128)))
                              bind(v2_3, ((ax3_0_0_1*128) + floormod((((((ax0_ax1_ax2_fused_0_3*2048) + (ax0_ax1_ax2_fused_1_3*512)) + (ax0_ax1_ax2_fused_2_3*256)) + (ax0_ax1_ax2_fused_3_2*8)) + ax0_ax1_ax2_fused_4_2), 128)))
                              tir.reads([C_intermediate[v1_3, v2_3]])
                              tir.writes([A_reindex_shared_dyn_1[v0_3, v1_3, v2_3]])
                              tir.attrs({"permuted_layout": 0, "buffer_dim_align": [[0, 1, 16, 8]]})
                              A_reindex_shared_dyn_1[v0_3, v1_3, v2_3] = C_intermediate[v1_3, v2_3]
                          }
                        }
                      }
                    }
                  }
                  for (ax0_ax1_ax2_fused_0_4: int32, 0, 1) "thread_binding" {
                    for (ax0_ax1_ax2_fused_1_4: int32, 0, 4) "thread_binding" {
                      for (ax0_ax1_ax2_fused_2_4: int32, 0, 16) "unroll" {
                        for (ax0_ax1_ax2_fused_3_3: int32, 0, 32) "thread_binding" {
                          for (ax0_ax1_ax2_fused_4_3: int32, 0, 8) "vectorized" {
                            block([1, 512, 128], "B_reindex_shared.dyn_1") as [v0_4, v1_4, v2_4] {
                              bind(v0_4, 0)
                              bind(v1_4, ((ax3_0_0_1*128) + floordiv((((((ax0_ax1_ax2_fused_0_4*16384) + (ax0_ax1_ax2_fused_1_4*4096)) + (ax0_ax1_ax2_fused_2_4*256)) + (ax0_ax1_ax2_fused_3_3*8)) + ax0_ax1_ax2_fused_4_3), 128)))
                              bind(v2_4, floormod((((((ax0_ax1_ax2_fused_0_4*16384) + (ax0_ax1_ax2_fused_1_4*4096)) + (ax0_ax1_ax2_fused_2_4*256)) + (ax0_ax1_ax2_fused_3_3*8)) + ax0_ax1_ax2_fused_4_3), 128))
                              tir.reads([D[v1_4, v2_4]])
                              tir.writes([B_reindex_shared_dyn_1[v0_4, v1_4, v2_4]])
                              tir.attrs({"permuted_layout": 0, "buffer_dim_align": [[0, 1, 16, 8]]})
                              B_reindex_shared_dyn_1[v0_4, v1_4, v2_4] = D[v1_4, v2_4]
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
                          tir.reads([A_reindex_shared_dyn_1[v0_o_6, (v1_o_6*16):((v1_o_6*16) + 16), (v2_o_6*16):((v2_o_6*16) + 16)]])
                          tir.writes([A_reindex_shared_dyn_warp_1[v0_o_6, v1_o_6, v2_o_6, 0:32, 0:8]])
                          tir.attrs({"permuted_layout": 0})
                          warp_4 = match_buffer(A_reindex_shared_dyn_warp_1[v0_o_6, v1_o_6, v2_o_6, 0:32, 0:8])
                          shared_4 = match_buffer(A_reindex_shared_dyn_1[v0_o_6, (v1_o_6*16):((v1_o_6*16) + 16), (v2_o_6*16):((v2_o_6*16) + 16)])
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
                          tir.reads([C_reindex_shared_dyn_warp_1[0, v1_o_8, v2_o_8, 0:32, 0:8], A_reindex_shared_dyn_warp_1[0, v1_o_8, v3_o_1, 0:32, 0:8], B_reindex_shared_dyn_warp_1[0, v3_o_1, v2_o_8, 0:32, 0:8]])
                          tir.writes([C_reindex_shared_dyn_warp_1[0, v1_o_8, v2_o_8, 0:32, 0:8]])
                          block([1, 1, tir.reduce_axis(0, 1)], "gemm_o_1") as [v1_i_o_1, v2_i_o_1, v3_i_o_1] {
                            bind(v1_i_o_1, 0)
                            bind(v2_i_o_1, 0)
                            bind(v3_i_o_1, 0)
                            tir.reads([C_reindex_shared_dyn_warp_1[0, v1_o_8, v2_o_8, 0:32, 0:8], A_reindex_shared_dyn_warp_1[0, v1_o_8, v3_o_1, 0:32, 0:8], B_reindex_shared_dyn_warp_1[0, v3_o_1, v2_o_8, 0:32, 0:8]])
                            tir.writes([C_reindex_shared_dyn_warp_1[0, v1_o_8, v2_o_8, 0:32, 0:8]])
                            A_1_3 = match_buffer(A_reindex_shared_dyn_warp_1[0, v1_o_8, v3_o_1, 0:32, 0:8])
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
              for (ax0_ax1_ax2_fused_0_5: int32, 0, 8) "unroll" {
                for (ax0_ax1_ax2_fused_1_5: int32, 0, 32) "thread_binding" {
                  for (ax0_ax1_ax2_fused_2_5: int32, 0, 8) "vectorized" {
                    block([1, 512, 128], "C_reindex_shared.dyn_1") as [v0_5, v1_5, v2_5] {
                      bind(v0_5, 0)
                      bind(v1_5, ((ax1_0_0_ax2_0_0_fused_1*16) + floordiv((((ax0_ax1_ax2_fused_0_5*256) + (ax0_ax1_ax2_fused_1_5*8)) + ax0_ax1_ax2_fused_2_5), 128)))
                      bind(v2_5, floormod((((ax0_ax1_ax2_fused_0_5*256) + (ax0_ax1_ax2_fused_1_5*8)) + ax0_ax1_ax2_fused_2_5), 128))
                      tir.reads([C_reindex_shared_dyn_1[v0_5, v1_5, v2_5]])
                      tir.writes([C_intermediate_1[v1_5, v2_5]])
                      C_intermediate_1[v1_5, v2_5] = C_reindex_shared_dyn_1[v0_5, v1_5, v2_5]
                  }
                }
              }
            }
          }
        }
      }
      for (ax0_2: int32, 0, 1) "thread_binding" {
        for (ax1_0_0_ax2_0_0_fused_2: int32, 0, 32) "thread_binding" {
          for (ax1_0_1_ax2_0_1_fused_2: int32, 0, 1) "thread_binding" {
            for (ax1_0_2_2: int32, 0, 1) "thread_binding" {
              for (ax2_0_2_2: int32, 0, 4) "thread_binding" {
                for (ax1_0_3_init_2: int32, 0, 1) {
                  for (ax2_0_3_init_2: int32, 0, 2) {
                    block([1, 32, 8], "gemm_o_init_2") as [v0_o_10, v1_o_10, v2_o_10] {
                      bind(v0_o_10, ax0_2)
                      bind(v1_o_10, ((ax1_0_0_ax2_0_0_fused_2 + ax1_0_2_2) + ax1_0_3_init_2))
                      bind(v2_o_10, ((ax2_0_2_2*2) + ax2_0_3_init_2))
                      tir.reads([])
                      tir.writes([C_reindex_shared_dyn_warp_2[0, v1_o_10, v2_o_10, 0:32, 0:8]])
                      block([1, 1], "gemm_init_o_2") as [v1_i_init_o_2, v2_i_init_o_2] {
                        bind(v1_i_init_o_2, 0)
                        bind(v2_i_init_o_2, 0)
                        tir.reads([])
                        tir.writes([C_reindex_shared_dyn_warp_2[0, v1_o_10, v2_o_10, 0:32, 0:8]])
                        C_warp_8 = match_buffer(C_reindex_shared_dyn_warp_2[0, v1_o_10, v2_o_10, 0:32, 0:8])
                        for (tx_10: int32, 0, 32) "thread_binding" {
                          @tir.mma_fill(8, C_warp_9: Pointer(warp float16), elem_offset_20: int32, dtype=float16)
                        }
                  }
                }
                for (ax3_0_0_2: int32, 0, 1) {
                  for (ax0_ax1_ax2_fused_0_6: int32, 0, 1) "thread_binding" {
                    for (ax0_ax1_ax2_fused_1_6: int32, 0, 4) "thread_binding" {
                      for (ax0_ax1_ax2_fused_2_6: int32, 0, 2) "unroll" {
                        for (ax0_ax1_ax2_fused_3_4: int32, 0, 32) "thread_binding" {
                          for (ax0_ax1_ax2_fused_4_4: int32, 0, 8) "vectorized" {
                            block([1, 512, 128], "A_reindex_shared.dyn_2") as [v0_6, v1_6, v2_6] {
                              bind(v0_6, 0)
                              bind(v1_6, ((ax1_0_0_ax2_0_0_fused_2*16) + floordiv((((((ax0_ax1_ax2_fused_0_6*2048) + (ax0_ax1_ax2_fused_1_6*512)) + (ax0_ax1_ax2_fused_2_6*256)) + (ax0_ax1_ax2_fused_3_4*8)) + ax0_ax1_ax2_fused_4_4), 128)))
                              bind(v2_6, floormod((((((ax0_ax1_ax2_fused_0_6*2048) + (ax0_ax1_ax2_fused_1_6*512)) + (ax0_ax1_ax2_fused_2_6*256)) + (ax0_ax1_ax2_fused_3_4*8)) + ax0_ax1_ax2_fused_4_4), 128))
                              tir.reads([C_intermediate_1[v1_6, v2_6]])
                              tir.writes([A_reindex_shared_dyn_2[v0_6, v1_6, v2_6]])
                              tir.attrs({"permuted_layout": 0, "buffer_dim_align": [[0, 1, 16, 8]]})
                              A_reindex_shared_dyn_2[v0_6, v1_6, v2_6] = C_intermediate_1[v1_6, v2_6]
                          }
                        }
                      }
                    }
                  }
                  for (ax0_ax1_ax2_fused_0_7: int32, 0, 1) "thread_binding" {
                    for (ax0_ax1_ax2_fused_1_7: int32, 0, 4) "thread_binding" {
                      for (ax0_ax1_ax2_fused_2_7: int32, 0, 16) "unroll" {
                        for (ax0_ax1_ax2_fused_3_5: int32, 0, 32) "thread_binding" {
                          for (ax0_ax1_ax2_fused_4_5: int32, 0, 8) "vectorized" {
                            block([1, 128, 128], "B_reindex_shared.dyn_2") as [v0_7, v1_7, v2_7] {
                              bind(v0_7, 0)
                              bind(v1_7, floordiv((((((ax0_ax1_ax2_fused_0_7*16384) + (ax0_ax1_ax2_fused_1_7*4096)) + (ax0_ax1_ax2_fused_2_7*256)) + (ax0_ax1_ax2_fused_3_5*8)) + ax0_ax1_ax2_fused_4_5), 128))
                              bind(v2_7, floormod((((((ax0_ax1_ax2_fused_0_7*16384) + (ax0_ax1_ax2_fused_1_7*4096)) + (ax0_ax1_ax2_fused_2_7*256)) + (ax0_ax1_ax2_fused_3_5*8)) + ax0_ax1_ax2_fused_4_5), 128))
                              tir.reads([F[v1_7, v2_7]])
                              tir.writes([B_reindex_shared_dyn_2[v0_7, v1_7, v2_7]])
                              tir.attrs({"permuted_layout": 0, "buffer_dim_align": [[0, 1, 16, 8]]})
                              B_reindex_shared_dyn_2[v0_7, v1_7, v2_7] = F[v1_7, v2_7]
                          }
                        }
                      }
                    }
                  }
                  for (ax3_0_1_2: int32, 0, 8) {
                    for (ax0_0_6: int32, 0, 1) {
                      for (ax1_0_8: int32, 0, 1) {
                        block([1, 32, 8], "A_reindex_shared.dyn_warp_o_2") as [v0_o_11, v1_o_11, v2_o_11] {
                          bind(v0_o_11, 0)
                          bind(v1_o_11, (ax1_0_0_ax2_0_0_fused_2 + ax0_0_6))
                          bind(v2_o_11, (ax3_0_1_2 + ax1_0_8))
                          tir.reads([A_reindex_shared_dyn_2[v0_o_11, (v1_o_11*16):((v1_o_11*16) + 16), (v2_o_11*16):((v2_o_11*16) + 16)]])
                          tir.writes([A_reindex_shared_dyn_warp_2[v0_o_11, v1_o_11, v2_o_11, 0:32, 0:8]])
                          tir.attrs({"permuted_layout": 0})
                          warp_8 = match_buffer(A_reindex_shared_dyn_warp_2[v0_o_11, v1_o_11, v2_o_11, 0:32, 0:8])
                          shared_8 = match_buffer(A_reindex_shared_dyn_2[v0_o_11, (v1_o_11*16):((v1_o_11*16) + 16), (v2_o_11*16):((v2_o_11*16) + 16)])
                          for (tx_11: int32, 0, 32) "thread_binding" {
                            @tir.ptx_ldmatrix(False, 4, ".b16", warp_9: Pointer(warp float16), (elem_offset_21: int32 + (8*tx_11)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_9: Pointer(shared.dyn float16), elem_offset_22: int32, (shared_s0_4: int32*16), 1, dtype=handle), ((shared_s0_4*floormod(tx_11, 16)) + (8*floordiv(tx_11, 16))), dtype=float16)
                          }
                      }
                    }
                    for (ax0_0_7: int32, 0, 1) {
                      for (ax1_0_9: int32, 0, 2) {
                        block([1, 8, 8], "B_reindex_shared.dyn_warp_o_2") as [v0_o_12, v1_o_12, v2_o_12] {
                          bind(v0_o_12, 0)
                          bind(v1_o_12, (ax3_0_1_2 + ax0_0_7))
                          bind(v2_o_12, ((ax2_0_2_2*2) + ax1_0_9))
                          tir.reads([B_reindex_shared_dyn_2[v0_o_12, (v1_o_12*16):((v1_o_12*16) + 16), (v2_o_12*16):((v2_o_12*16) + 16)]])
                          tir.writes([B_reindex_shared_dyn_warp_2[v0_o_12, v1_o_12, v2_o_12, 0:32, 0:8]])
                          tir.attrs({"permuted_layout": 0})
                          warp_10 = match_buffer(B_reindex_shared_dyn_warp_2[v0_o_12, v1_o_12, v2_o_12, 0:32, 0:8])
                          shared_10 = match_buffer(B_reindex_shared_dyn_2[v0_o_12, (v1_o_12*16):((v1_o_12*16) + 16), (v2_o_12*16):((v2_o_12*16) + 16)])
                          for (tx_12: int32, 0, 32) "thread_binding" {
                            @tir.ptx_ldmatrix(True, 4, ".b16", warp_11: Pointer(warp float16), (elem_offset_23: int32 + (8*tx_12)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_11: Pointer(shared.dyn float16), elem_offset_24: int32, (shared_s0_5: int32*16), 1, dtype=handle), ((shared_s0_5*floormod(tx_12, 16)) + (8*floordiv(tx_12, 16))), dtype=float16)
                          }
                      }
                    }
                    for (ax1_0_3_2: int32, 0, 1) {
                      for (ax2_0_3_2: int32, 0, 2) {
                        block([1, 32, 8, tir.reduce_axis(0, 8)], "gemm_o_update_2") as [v0_o_13, v1_o_13, v2_o_13, v3_o_2] {
                          bind(v0_o_13, ax0_2)
                          bind(v1_o_13, ((ax1_0_0_ax2_0_0_fused_2 + ax1_0_2_2) + ax1_0_3_2))
                          bind(v2_o_13, ((ax2_0_2_2*2) + ax2_0_3_2))
                          bind(v3_o_2, ((ax3_0_0_2*8) + ax3_0_1_2))
                          tir.reads([C_reindex_shared_dyn_warp_2[0, v1_o_13, v2_o_13, 0:32, 0:8], A_reindex_shared_dyn_warp_2[0, v1_o_13, v3_o_2, 0:32, 0:8], B_reindex_shared_dyn_warp_2[0, v3_o_2, v2_o_13, 0:32, 0:8]])
                          tir.writes([C_reindex_shared_dyn_warp_2[0, v1_o_13, v2_o_13, 0:32, 0:8]])
                          block([1, 1, tir.reduce_axis(0, 1)], "gemm_o_2") as [v1_i_o_2, v2_i_o_2, v3_i_o_2] {
                            bind(v1_i_o_2, 0)
                            bind(v2_i_o_2, 0)
                            bind(v3_i_o_2, 0)
                            tir.reads([C_reindex_shared_dyn_warp_2[0, v1_o_13, v2_o_13, 0:32, 0:8], A_reindex_shared_dyn_warp_2[0, v1_o_13, v3_o_2, 0:32, 0:8], B_reindex_shared_dyn_warp_2[0, v3_o_2, v2_o_13, 0:32, 0:8]])
                            tir.writes([C_reindex_shared_dyn_warp_2[0, v1_o_13, v2_o_13, 0:32, 0:8]])
                            A_1_5 = match_buffer(A_reindex_shared_dyn_warp_2[0, v1_o_13, v3_o_2, 0:32, 0:8])
                            B_1_5 = match_buffer(B_reindex_shared_dyn_warp_2[0, v3_o_2, v2_o_13, 0:32, 0:8])
                            C_1_8 = match_buffer(C_reindex_shared_dyn_warp_2[0, v1_o_13, v2_o_13, 0:32, 0:8])
                            for (tx_13: int32, 0, 32) "thread_binding" {
                              @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1_6: Pointer(warp float16), (elem_offset_25: int32 + (tx_13*8)), B_1_6: Pointer(warp float16), (elem_offset_26: int32 + (tx_13*8)), C_1_9: Pointer(warp float16), (elem_offset_27: int32 + (tx_13*8)), False, dtype=float16)
                              @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1_6, (elem_offset_25 + (tx_13*8)), B_1_6, ((elem_offset_26 + (tx_13*8)) + 4), C_1_9, ((elem_offset_27 + (tx_13*8)) + 4), False, dtype=float16)
                            }
                      }
                    }
                  }
                }
                for (ax0_0_8: int32, 0, 1) {
                  for (ax1_0_10: int32, 0, 2) {
                    block([1, 32, 8], "C_reindex_shared.dyn_warp_o_2") as [v0_o_14, v1_o_14, v2_o_14] {
                      bind(v0_o_14, 0)
                      bind(v1_o_14, ax1_0_0_ax2_0_0_fused_2)
                      bind(v2_o_14, ((ax2_0_2_2*2) + ax1_0_10))
                      tir.reads([C_reindex_shared_dyn_warp_2[v0_o_14, v1_o_14, v2_o_14, 0:32, 0:8]])
                      tir.writes([C_reindex_shared_dyn_2[v0_o_14, (v1_o_14*16):((v1_o_14*16) + 16), (v2_o_14*16):((v2_o_14*16) + 16)]])
                      C_warp_10 = match_buffer(C_reindex_shared_dyn_warp_2[v0_o_14, v1_o_14, v2_o_14, 0:32, 0:8])
                      C_1_10 = match_buffer(C_reindex_shared_dyn_2[v0_o_14, (v1_o_14*16):((v1_o_14*16) + 16), (v2_o_14*16):((v2_o_14*16) + 16)])
                      for (tx_14: int32, 0, 32) "thread_binding" {
                        @tir.mma_store(16, 16, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), C_1_11: Pointer(shared.dyn float16), elem_offset_28: int32, (C_1_s0_2: int32*16), 2, dtype=handle), C_warp_11: Pointer(warp float16), elem_offset_29: int32, C_1_s0_2, dtype=float16)
                      }
                  }
                }
              }
              for (ax0_ax1_ax2_fused_0_8: int32, 0, 8) "unroll" {
                for (ax0_ax1_ax2_fused_1_8: int32, 0, 32) "thread_binding" {
                  for (ax0_ax1_ax2_fused_2_8: int32, 0, 8) "vectorized" {
                    block([1, 512, 128], "C_reindex_shared.dyn_2") as [v0_8, v1_8, v2_8] {
                      bind(v0_8, 0)
                      bind(v1_8, ((ax1_0_0_ax2_0_0_fused_2*16) + floordiv((((ax0_ax1_ax2_fused_0_8*256) + (ax0_ax1_ax2_fused_1_8*8)) + ax0_ax1_ax2_fused_2_8), 128)))
                      bind(v2_8, floormod((((ax0_ax1_ax2_fused_0_8*256) + (ax0_ax1_ax2_fused_1_8*8)) + ax0_ax1_ax2_fused_2_8), 128))
                      tir.reads([C_reindex_shared_dyn_2[v0_8, v1_8, v2_8]])
                      tir.writes([C_intermediate_1_2[v1_8, v2_8]])
                      C_intermediate_1_2[v1_8, v2_8] = C_reindex_shared_dyn_2[v0_8, v1_8, v2_8]
                  }
                }
              }
            }
          }
        }
      }
    }
}

@gemm_0 = primfn(A_handle: handle, B_handle: handle, C_handle: handle) -> ()
  attr = {"dlight.tensorcore_prenormlized": True, "global_symbol": "gemm_0", "op_pattern": 0}
  buffers = {A_2: Buffer(A_3: Pointer(global float16), float16, [512, 128], []),
             B_2: Buffer(B_3: Pointer(global float16), float16, [128, 512], []),
             C_2: Buffer(C_3: Pointer(global float16), float16, [512, 512], [])}
  buffer_map = {A_handle: A_2, B_handle: B_2, C_handle: C_2} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_reindex_shared_dyn_3 = alloc_buffer(float16[1, 512, 128])
    B_reindex_shared_dyn_3 = alloc_buffer(float16[1, 128, 512])
    A_reindex_shared_dyn_warp_3 = alloc_buffer(float16[1, 32, 8, 32, 8])
    B_reindex_shared_dyn_warp_3 = alloc_buffer(float16[1, 8, 32, 32, 8])
    C_reindex_shared_dyn_3 = alloc_buffer(float16[1, 512, 512])
    C_reindex_shared_dyn_warp_3 = alloc_buffer(float16[1, 32, 32, 32, 8])
    for (ax0_3: int32, 0, 1) "thread_binding" {
      for (ax1_0_0_ax2_0_0_fused_3: int32, 0, 32) "thread_binding" {
        for (ax1_0_1_ax2_0_1_fused_3: int32, 0, 1) "thread_binding" {
          for (ax1_0_2_3: int32, 0, 1) "thread_binding" {
            for (ax2_0_2_3: int32, 0, 4) "thread_binding" {
              for (ax1_0_3_init_3: int32, 0, 1) {
                for (ax2_0_3_init_3: int32, 0, 8) {
                  block([1, 32, 32], "gemm_o_init") as [v0_o_15, v1_o_15, v2_o_15] {
                    bind(v0_o_15, ax0_3)
                    bind(v1_o_15, ((ax1_0_0_ax2_0_0_fused_3 + ax1_0_2_3) + ax1_0_3_init_3))
                    bind(v2_o_15, ((ax2_0_2_3*8) + ax2_0_3_init_3))
                    tir.reads([])
                    tir.writes([C_reindex_shared_dyn_warp_3[0, v1_o_15, v2_o_15, 0:32, 0:8]])
                    block([1, 1], "gemm_init_o") as [v1_i_init_o_3, v2_i_init_o_3] {
                      bind(v1_i_init_o_3, 0)
                      bind(v2_i_init_o_3, 0)
                      tir.reads([])
                      tir.writes([C_reindex_shared_dyn_warp_3[0, v1_o_15, v2_o_15, 0:32, 0:8]])
                      C_warp_12 = match_buffer(C_reindex_shared_dyn_warp_3[0, v1_o_15, v2_o_15, 0:32, 0:8])
                      for (tx_15: int32, 0, 32) "thread_binding" {
                        @tir.mma_fill(8, C_warp_13: Pointer(warp float16), elem_offset_30: int32, dtype=float16)
                      }
                }
              }
              for (ax3_0_0_3: int32, 0, 1) {
                for (ax0_ax1_ax2_fused_0_9: int32, 0, 1) "thread_binding" {
                  for (ax0_ax1_ax2_fused_1_9: int32, 0, 4) "thread_binding" {
                    for (ax0_ax1_ax2_fused_2_9: int32, 0, 2) "unroll" {
                      for (ax0_ax1_ax2_fused_3_6: int32, 0, 32) "thread_binding" {
                        for (ax0_ax1_ax2_fused_4_6: int32, 0, 8) "vectorized" {
                          block([1, 512, 128], "A_reindex_shared.dyn") as [v0_9, v1_9, v2_9] {
                            bind(v0_9, 0)
                            bind(v1_9, ((ax1_0_0_ax2_0_0_fused_3*16) + floordiv((((((ax0_ax1_ax2_fused_0_9*2048) + (ax0_ax1_ax2_fused_1_9*512)) + (ax0_ax1_ax2_fused_2_9*256)) + (ax0_ax1_ax2_fused_3_6*8)) + ax0_ax1_ax2_fused_4_6), 128)))
                            bind(v2_9, floormod((((((ax0_ax1_ax2_fused_0_9*2048) + (ax0_ax1_ax2_fused_1_9*512)) + (ax0_ax1_ax2_fused_2_9*256)) + (ax0_ax1_ax2_fused_3_6*8)) + ax0_ax1_ax2_fused_4_6), 128))
                            tir.reads([A_2[v1_9, v2_9]])
                            tir.writes([A_reindex_shared_dyn_3[v0_9, v1_9, v2_9]])
                            tir.attrs({"permuted_layout": 0, "buffer_dim_align": [[0, 1, 16, 8]]})
                            A_reindex_shared_dyn_3[v0_9, v1_9, v2_9] = A_2[v1_9, v2_9]
                        }
                      }
                    }
                  }
                }
                for (ax0_ax1_ax2_fused_0_10: int32, 0, 1) "thread_binding" {
                  for (ax0_ax1_ax2_fused_1_10: int32, 0, 4) "thread_binding" {
                    for (ax0_ax1_ax2_fused_2_10: int32, 0, 64) "unroll" {
                      for (ax0_ax1_ax2_fused_3_7: int32, 0, 32) "thread_binding" {
                        for (ax0_ax1_ax2_fused_4_7: int32, 0, 8) "vectorized" {
                          block([1, 128, 512], "B_reindex_shared.dyn") as [v0_10, v1_10, v2_10] {
                            bind(v0_10, 0)
                            bind(v1_10, floordiv((((((ax0_ax1_ax2_fused_0_10*65536) + (ax0_ax1_ax2_fused_1_10*16384)) + (ax0_ax1_ax2_fused_2_10*256)) + (ax0_ax1_ax2_fused_3_7*8)) + ax0_ax1_ax2_fused_4_7), 512))
                            bind(v2_10, floormod((((((ax0_ax1_ax2_fused_0_10*65536) + (ax0_ax1_ax2_fused_1_10*16384)) + (ax0_ax1_ax2_fused_2_10*256)) + (ax0_ax1_ax2_fused_3_7*8)) + ax0_ax1_ax2_fused_4_7), 512))
                            tir.reads([B_2[v1_10, v2_10]])
                            tir.writes([B_reindex_shared_dyn_3[v0_10, v1_10, v2_10]])
                            tir.attrs({"permuted_layout": 0, "buffer_dim_align": [[0, 1, 16, 8]]})
                            B_reindex_shared_dyn_3[v0_10, v1_10, v2_10] = B_2[v1_10, v2_10]
                        }
                      }
                    }
                  }
                }
                for (ax3_0_1_3: int32, 0, 8) {
                  for (ax0_0_9: int32, 0, 1) {
                    for (ax1_0_11: int32, 0, 1) {
                      block([1, 32, 8], "A_reindex_shared.dyn_warp_o") as [v0_o_16, v1_o_16, v2_o_16] {
                        bind(v0_o_16, 0)
                        bind(v1_o_16, (ax1_0_0_ax2_0_0_fused_3 + ax0_0_9))
                        bind(v2_o_16, (ax3_0_1_3 + ax1_0_11))
                        tir.reads([A_reindex_shared_dyn_3[v0_o_16, (v1_o_16*16):((v1_o_16*16) + 16), (v2_o_16*16):((v2_o_16*16) + 16)]])
                        tir.writes([A_reindex_shared_dyn_warp_3[v0_o_16, v1_o_16, v2_o_16, 0:32, 0:8]])
                        tir.attrs({"permuted_layout": 0})
                        warp_12 = match_buffer(A_reindex_shared_dyn_warp_3[v0_o_16, v1_o_16, v2_o_16, 0:32, 0:8])
                        shared_12 = match_buffer(A_reindex_shared_dyn_3[v0_o_16, (v1_o_16*16):((v1_o_16*16) + 16), (v2_o_16*16):((v2_o_16*16) + 16)])
                        for (tx_16: int32, 0, 32) "thread_binding" {
                          @tir.ptx_ldmatrix(False, 4, ".b16", warp_13: Pointer(warp float16), (elem_offset_31: int32 + (8*tx_16)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_13: Pointer(shared.dyn float16), elem_offset_32: int32, (shared_s0_6: int32*16), 1, dtype=handle), ((shared_s0_6*floormod(tx_16, 16)) + (8*floordiv(tx_16, 16))), dtype=float16)
                        }
                    }
                  }
                  for (ax0_0_10: int32, 0, 1) {
                    for (ax1_0_12: int32, 0, 8) {
                      block([1, 8, 32], "B_reindex_shared.dyn_warp_o") as [v0_o_17, v1_o_17, v2_o_17] {
                        bind(v0_o_17, 0)
                        bind(v1_o_17, (ax3_0_1_3 + ax0_0_10))
                        bind(v2_o_17, ((ax2_0_2_3*8) + ax1_0_12))
                        tir.reads([B_reindex_shared_dyn_3[v0_o_17, (v1_o_17*16):((v1_o_17*16) + 16), (v2_o_17*16):((v2_o_17*16) + 16)]])
                        tir.writes([B_reindex_shared_dyn_warp_3[v0_o_17, v1_o_17, v2_o_17, 0:32, 0:8]])
                        tir.attrs({"permuted_layout": 0})
                        warp_14 = match_buffer(B_reindex_shared_dyn_warp_3[v0_o_17, v1_o_17, v2_o_17, 0:32, 0:8])
                        shared_14 = match_buffer(B_reindex_shared_dyn_3[v0_o_17, (v1_o_17*16):((v1_o_17*16) + 16), (v2_o_17*16):((v2_o_17*16) + 16)])
                        for (tx_17: int32, 0, 32) "thread_binding" {
                          @tir.ptx_ldmatrix(True, 4, ".b16", warp_15: Pointer(warp float16), (elem_offset_33: int32 + (8*tx_17)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_15: Pointer(shared.dyn float16), elem_offset_34: int32, (shared_s0_7: int32*16), 1, dtype=handle), ((shared_s0_7*floormod(tx_17, 16)) + (8*floordiv(tx_17, 16))), dtype=float16)
                        }
                    }
                  }
                  for (ax1_0_3_3: int32, 0, 1) {
                    for (ax2_0_3_3: int32, 0, 8) {
                      block([1, 32, 32, tir.reduce_axis(0, 8)], "gemm_o_update") as [v0_o_18, v1_o_18, v2_o_18, v3_o_3] {
                        bind(v0_o_18, ax0_3)
                        bind(v1_o_18, ((ax1_0_0_ax2_0_0_fused_3 + ax1_0_2_3) + ax1_0_3_3))
                        bind(v2_o_18, ((ax2_0_2_3*8) + ax2_0_3_3))
                        bind(v3_o_3, ((ax3_0_0_3*8) + ax3_0_1_3))
                        tir.reads([C_reindex_shared_dyn_warp_3[0, v1_o_18, v2_o_18, 0:32, 0:8], A_reindex_shared_dyn_warp_3[0, v1_o_18, v3_o_3, 0:32, 0:8], B_reindex_shared_dyn_warp_3[0, v3_o_3, v2_o_18, 0:32, 0:8]])
                        tir.writes([C_reindex_shared_dyn_warp_3[0, v1_o_18, v2_o_18, 0:32, 0:8]])
                        block([1, 1, tir.reduce_axis(0, 1)], "gemm_o") as [v1_i_o_3, v2_i_o_3, v3_i_o_3] {
                          bind(v1_i_o_3, 0)
                          bind(v2_i_o_3, 0)
                          bind(v3_i_o_3, 0)
                          tir.reads([C_reindex_shared_dyn_warp_3[0, v1_o_18, v2_o_18, 0:32, 0:8], A_reindex_shared_dyn_warp_3[0, v1_o_18, v3_o_3, 0:32, 0:8], B_reindex_shared_dyn_warp_3[0, v3_o_3, v2_o_18, 0:32, 0:8]])
                          tir.writes([C_reindex_shared_dyn_warp_3[0, v1_o_18, v2_o_18, 0:32, 0:8]])
                          A_1_7 = match_buffer(A_reindex_shared_dyn_warp_3[0, v1_o_18, v3_o_3, 0:32, 0:8])
                          B_1_7 = match_buffer(B_reindex_shared_dyn_warp_3[0, v3_o_3, v2_o_18, 0:32, 0:8])
                          C_1_12 = match_buffer(C_reindex_shared_dyn_warp_3[0, v1_o_18, v2_o_18, 0:32, 0:8])
                          for (tx_18: int32, 0, 32) "thread_binding" {
                            @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1_8: Pointer(warp float16), (elem_offset_35: int32 + (tx_18*8)), B_1_8: Pointer(warp float16), (elem_offset_36: int32 + (tx_18*8)), C_1_13: Pointer(warp float16), (elem_offset_37: int32 + (tx_18*8)), False, dtype=float16)
                            @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1_8, (elem_offset_35 + (tx_18*8)), B_1_8, ((elem_offset_36 + (tx_18*8)) + 4), C_1_13, ((elem_offset_37 + (tx_18*8)) + 4), False, dtype=float16)
                          }
                    }
                  }
                }
              }
              for (ax0_0_11: int32, 0, 1) {
                for (ax1_0_13: int32, 0, 8) {
                  block([1, 32, 32], "C_reindex_shared.dyn_warp_o") as [v0_o_19, v1_o_19, v2_o_19] {
                    bind(v0_o_19, 0)
                    bind(v1_o_19, ax1_0_0_ax2_0_0_fused_3)
                    bind(v2_o_19, ((ax2_0_2_3*8) + ax1_0_13))
                    tir.reads([C_reindex_shared_dyn_warp_3[v0_o_19, v1_o_19, v2_o_19, 0:32, 0:8]])
                    tir.writes([C_reindex_shared_dyn_3[v0_o_19, (v1_o_19*16):((v1_o_19*16) + 16), (v2_o_19*16):((v2_o_19*16) + 16)]])
                    C_warp_14 = match_buffer(C_reindex_shared_dyn_warp_3[v0_o_19, v1_o_19, v2_o_19, 0:32, 0:8])
                    C_1_14 = match_buffer(C_reindex_shared_dyn_3[v0_o_19, (v1_o_19*16):((v1_o_19*16) + 16), (v2_o_19*16):((v2_o_19*16) + 16)])
                    for (tx_19: int32, 0, 32) "thread_binding" {
                      @tir.mma_store(16, 16, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), C_1_15: Pointer(shared.dyn float16), elem_offset_38: int32, (C_1_s0_3: int32*16), 2, dtype=handle), C_warp_15: Pointer(warp float16), elem_offset_39: int32, C_1_s0_3, dtype=float16)
                    }
                }
              }
            }
            for (ax0_ax1_ax2_fused_0_11: int32, 0, 32) "unroll" {
              for (ax0_ax1_ax2_fused_1_11: int32, 0, 32) "thread_binding" {
                for (ax0_ax1_ax2_fused_2_11: int32, 0, 8) "vectorized" {
                  block([1, 512, 512], "C_reindex_shared.dyn") as [v0_11, v1_11, v2_11] {
                    bind(v0_11, 0)
                    bind(v1_11, ((ax1_0_0_ax2_0_0_fused_3*16) + floordiv((((ax0_ax1_ax2_fused_0_11*256) + (ax0_ax1_ax2_fused_1_11*8)) + ax0_ax1_ax2_fused_2_11), 512)))
                    bind(v2_11, floormod((((ax0_ax1_ax2_fused_0_11*256) + (ax0_ax1_ax2_fused_1_11*8)) + ax0_ax1_ax2_fused_2_11), 512))
                    tir.reads([C_reindex_shared_dyn_3[v0_11, v1_11, v2_11]])
                    tir.writes([C_2[v1_11, v2_11]])
                    C_2[v1_11, v2_11] = C_reindex_shared_dyn_3[v0_11, v1_11, v2_11]
                }
              }
            }
          }
        }
      }
    }
}

@gemm_1 = primfn(A_handle_1: handle, B_handle_1: handle, C_handle_1: handle) -> ()
  attr = {"dlight.tensorcore_prenormlized": True, "global_symbol": "gemm_1", "op_pattern": 0}
  buffers = {A_4: Buffer(A_5: Pointer(global float16), float16, [512, 512], []),
             B_4: Buffer(B_5: Pointer(global float16), float16, [512, 128], []),
             C_4: Buffer(C_5: Pointer(global float16), float16, [512, 128], [])}
  buffer_map = {A_handle_1: A_4, B_handle_1: B_4, C_handle_1: C_4} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_reindex_shared_dyn_4 = alloc_buffer(float16[1, 512, 512])
    B_reindex_shared_dyn_4 = alloc_buffer(float16[1, 512, 128])
    A_reindex_shared_dyn_warp_4 = alloc_buffer(float16[1, 32, 32, 32, 8])
    B_reindex_shared_dyn_warp_4 = alloc_buffer(float16[1, 32, 8, 32, 8])
    C_reindex_shared_dyn_4 = alloc_buffer(float16[1, 512, 128])
    C_reindex_shared_dyn_warp_4 = alloc_buffer(float16[1, 32, 8, 32, 8])
    for (ax0_4: int32, 0, 1) "thread_binding" {
      for (ax1_0_0_ax2_0_0_fused_4: int32, 0, 32) "thread_binding" {
        for (ax1_0_1_ax2_0_1_fused_4: int32, 0, 1) "thread_binding" {
          for (ax1_0_2_4: int32, 0, 1) "thread_binding" {
            for (ax2_0_2_4: int32, 0, 4) "thread_binding" {
              for (ax1_0_3_init_4: int32, 0, 1) {
                for (ax2_0_3_init_4: int32, 0, 2) {
                  block([1, 32, 8], "gemm_o_init") as [v0_o_20, v1_o_20, v2_o_20] {
                    bind(v0_o_20, ax0_4)
                    bind(v1_o_20, ((ax1_0_0_ax2_0_0_fused_4 + ax1_0_2_4) + ax1_0_3_init_4))
                    bind(v2_o_20, ((ax2_0_2_4*2) + ax2_0_3_init_4))
                    tir.reads([])
                    tir.writes([C_reindex_shared_dyn_warp_4[0, v1_o_20, v2_o_20, 0:32, 0:8]])
                    block([1, 1], "gemm_init_o") as [v1_i_init_o_4, v2_i_init_o_4] {
                      bind(v1_i_init_o_4, 0)
                      bind(v2_i_init_o_4, 0)
                      tir.reads([])
                      tir.writes([C_reindex_shared_dyn_warp_4[0, v1_o_20, v2_o_20, 0:32, 0:8]])
                      C_warp_16 = match_buffer(C_reindex_shared_dyn_warp_4[0, v1_o_20, v2_o_20, 0:32, 0:8])
                      for (tx_20: int32, 0, 32) "thread_binding" {
                        @tir.mma_fill(8, C_warp_17: Pointer(warp float16), elem_offset_40: int32, dtype=float16)
                      }
                }
              }
              for (ax3_0_0_4: int32, 0, 4) {
                for (ax0_ax1_ax2_fused_0_12: int32, 0, 1) "thread_binding" {
                  for (ax0_ax1_ax2_fused_1_12: int32, 0, 4) "thread_binding" {
                    for (ax0_ax1_ax2_fused_2_12: int32, 0, 2) "unroll" {
                      for (ax0_ax1_ax2_fused_3_8: int32, 0, 32) "thread_binding" {
                        for (ax0_ax1_ax2_fused_4_8: int32, 0, 8) "vectorized" {
                          block([1, 512, 512], "A_reindex_shared.dyn") as [v0_12, v1_12, v2_12] {
                            bind(v0_12, 0)
                            bind(v1_12, ((ax1_0_0_ax2_0_0_fused_4*16) + floordiv((((((ax0_ax1_ax2_fused_0_12*2048) + (ax0_ax1_ax2_fused_1_12*512)) + (ax0_ax1_ax2_fused_2_12*256)) + (ax0_ax1_ax2_fused_3_8*8)) + ax0_ax1_ax2_fused_4_8), 128)))
                            bind(v2_12, ((ax3_0_0_4*128) + floormod((((((ax0_ax1_ax2_fused_0_12*2048) + (ax0_ax1_ax2_fused_1_12*512)) + (ax0_ax1_ax2_fused_2_12*256)) + (ax0_ax1_ax2_fused_3_8*8)) + ax0_ax1_ax2_fused_4_8), 128)))
                            tir.reads([A_4[v1_12, v2_12]])
                            tir.writes([A_reindex_shared_dyn_4[v0_12, v1_12, v2_12]])
                            tir.attrs({"permuted_layout": 0, "buffer_dim_align": [[0, 1, 16, 8]]})
                            A_reindex_shared_dyn_4[v0_12, v1_12, v2_12] = A_4[v1_12, v2_12]
                        }
                      }
                    }
                  }
                }
                for (ax0_ax1_ax2_fused_0_13: int32, 0, 1) "thread_binding" {
                  for (ax0_ax1_ax2_fused_1_13: int32, 0, 4) "thread_binding" {
                    for (ax0_ax1_ax2_fused_2_13: int32, 0, 16) "unroll" {
                      for (ax0_ax1_ax2_fused_3_9: int32, 0, 32) "thread_binding" {
                        for (ax0_ax1_ax2_fused_4_9: int32, 0, 8) "vectorized" {
                          block([1, 512, 128], "B_reindex_shared.dyn") as [v0_13, v1_13, v2_13] {
                            bind(v0_13, 0)
                            bind(v1_13, ((ax3_0_0_4*128) + floordiv((((((ax0_ax1_ax2_fused_0_13*16384) + (ax0_ax1_ax2_fused_1_13*4096)) + (ax0_ax1_ax2_fused_2_13*256)) + (ax0_ax1_ax2_fused_3_9*8)) + ax0_ax1_ax2_fused_4_9), 128)))
                            bind(v2_13, floormod((((((ax0_ax1_ax2_fused_0_13*16384) + (ax0_ax1_ax2_fused_1_13*4096)) + (ax0_ax1_ax2_fused_2_13*256)) + (ax0_ax1_ax2_fused_3_9*8)) + ax0_ax1_ax2_fused_4_9), 128))
                            tir.reads([B_4[v1_13, v2_13]])
                            tir.writes([B_reindex_shared_dyn_4[v0_13, v1_13, v2_13]])
                            tir.attrs({"permuted_layout": 0, "buffer_dim_align": [[0, 1, 16, 8]]})
                            B_reindex_shared_dyn_4[v0_13, v1_13, v2_13] = B_4[v1_13, v2_13]
                        }
                      }
                    }
                  }
                }
                for (ax3_0_1_4: int32, 0, 8) {
                  for (ax0_0_12: int32, 0, 1) {
                    for (ax1_0_14: int32, 0, 1) {
                      block([1, 32, 32], "A_reindex_shared.dyn_warp_o") as [v0_o_21, v1_o_21, v2_o_21] {
                        bind(v0_o_21, 0)
                        bind(v1_o_21, (ax1_0_0_ax2_0_0_fused_4 + ax0_0_12))
                        bind(v2_o_21, (((ax3_0_0_4*8) + ax3_0_1_4) + ax1_0_14))
                        tir.reads([A_reindex_shared_dyn_4[v0_o_21, (v1_o_21*16):((v1_o_21*16) + 16), (v2_o_21*16):((v2_o_21*16) + 16)]])
                        tir.writes([A_reindex_shared_dyn_warp_4[v0_o_21, v1_o_21, v2_o_21, 0:32, 0:8]])
                        tir.attrs({"permuted_layout": 0})
                        warp_16 = match_buffer(A_reindex_shared_dyn_warp_4[v0_o_21, v1_o_21, v2_o_21, 0:32, 0:8])
                        shared_16 = match_buffer(A_reindex_shared_dyn_4[v0_o_21, (v1_o_21*16):((v1_o_21*16) + 16), (v2_o_21*16):((v2_o_21*16) + 16)])
                        for (tx_21: int32, 0, 32) "thread_binding" {
                          @tir.ptx_ldmatrix(False, 4, ".b16", warp_17: Pointer(warp float16), (elem_offset_41: int32 + (8*tx_21)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_17: Pointer(shared.dyn float16), elem_offset_42: int32, (shared_s0_8: int32*16), 1, dtype=handle), ((shared_s0_8*floormod(tx_21, 16)) + (8*floordiv(tx_21, 16))), dtype=float16)
                        }
                    }
                  }
                  for (ax0_0_13: int32, 0, 1) {
                    for (ax1_0_15: int32, 0, 2) {
                      block([1, 32, 8], "B_reindex_shared.dyn_warp_o") as [v0_o_22, v1_o_22, v2_o_22] {
                        bind(v0_o_22, 0)
                        bind(v1_o_22, (((ax3_0_0_4*8) + ax3_0_1_4) + ax0_0_13))
                        bind(v2_o_22, ((ax2_0_2_4*2) + ax1_0_15))
                        tir.reads([B_reindex_shared_dyn_4[v0_o_22, (v1_o_22*16):((v1_o_22*16) + 16), (v2_o_22*16):((v2_o_22*16) + 16)]])
                        tir.writes([B_reindex_shared_dyn_warp_4[v0_o_22, v1_o_22, v2_o_22, 0:32, 0:8]])
                        tir.attrs({"permuted_layout": 0})
                        warp_18 = match_buffer(B_reindex_shared_dyn_warp_4[v0_o_22, v1_o_22, v2_o_22, 0:32, 0:8])
                        shared_18 = match_buffer(B_reindex_shared_dyn_4[v0_o_22, (v1_o_22*16):((v1_o_22*16) + 16), (v2_o_22*16):((v2_o_22*16) + 16)])
                        for (tx_22: int32, 0, 32) "thread_binding" {
                          @tir.ptx_ldmatrix(True, 4, ".b16", warp_19: Pointer(warp float16), (elem_offset_43: int32 + (8*tx_22)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_19: Pointer(shared.dyn float16), elem_offset_44: int32, (shared_s0_9: int32*16), 1, dtype=handle), ((shared_s0_9*floormod(tx_22, 16)) + (8*floordiv(tx_22, 16))), dtype=float16)
                        }
                    }
                  }
                  for (ax1_0_3_4: int32, 0, 1) {
                    for (ax2_0_3_4: int32, 0, 2) {
                      block([1, 32, 8, tir.reduce_axis(0, 32)], "gemm_o_update") as [v0_o_23, v1_o_23, v2_o_23, v3_o_4] {
                        bind(v0_o_23, ax0_4)
                        bind(v1_o_23, ((ax1_0_0_ax2_0_0_fused_4 + ax1_0_2_4) + ax1_0_3_4))
                        bind(v2_o_23, ((ax2_0_2_4*2) + ax2_0_3_4))
                        bind(v3_o_4, ((ax3_0_0_4*8) + ax3_0_1_4))
                        tir.reads([C_reindex_shared_dyn_warp_4[0, v1_o_23, v2_o_23, 0:32, 0:8], A_reindex_shared_dyn_warp_4[0, v1_o_23, v3_o_4, 0:32, 0:8], B_reindex_shared_dyn_warp_4[0, v3_o_4, v2_o_23, 0:32, 0:8]])
                        tir.writes([C_reindex_shared_dyn_warp_4[0, v1_o_23, v2_o_23, 0:32, 0:8]])
                        block([1, 1, tir.reduce_axis(0, 1)], "gemm_o") as [v1_i_o_4, v2_i_o_4, v3_i_o_4] {
                          bind(v1_i_o_4, 0)
                          bind(v2_i_o_4, 0)
                          bind(v3_i_o_4, 0)
                          tir.reads([C_reindex_shared_dyn_warp_4[0, v1_o_23, v2_o_23, 0:32, 0:8], A_reindex_shared_dyn_warp_4[0, v1_o_23, v3_o_4, 0:32, 0:8], B_reindex_shared_dyn_warp_4[0, v3_o_4, v2_o_23, 0:32, 0:8]])
                          tir.writes([C_reindex_shared_dyn_warp_4[0, v1_o_23, v2_o_23, 0:32, 0:8]])
                          A_1_9 = match_buffer(A_reindex_shared_dyn_warp_4[0, v1_o_23, v3_o_4, 0:32, 0:8])
                          B_1_9 = match_buffer(B_reindex_shared_dyn_warp_4[0, v3_o_4, v2_o_23, 0:32, 0:8])
                          C_1_16 = match_buffer(C_reindex_shared_dyn_warp_4[0, v1_o_23, v2_o_23, 0:32, 0:8])
                          for (tx_23: int32, 0, 32) "thread_binding" {
                            @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1_10: Pointer(warp float16), (elem_offset_45: int32 + (tx_23*8)), B_1_10: Pointer(warp float16), (elem_offset_46: int32 + (tx_23*8)), C_1_17: Pointer(warp float16), (elem_offset_47: int32 + (tx_23*8)), False, dtype=float16)
                            @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1_10, (elem_offset_45 + (tx_23*8)), B_1_10, ((elem_offset_46 + (tx_23*8)) + 4), C_1_17, ((elem_offset_47 + (tx_23*8)) + 4), False, dtype=float16)
                          }
                    }
                  }
                }
              }
              for (ax0_0_14: int32, 0, 1) {
                for (ax1_0_16: int32, 0, 2) {
                  block([1, 32, 8], "C_reindex_shared.dyn_warp_o") as [v0_o_24, v1_o_24, v2_o_24] {
                    bind(v0_o_24, 0)
                    bind(v1_o_24, ax1_0_0_ax2_0_0_fused_4)
                    bind(v2_o_24, ((ax2_0_2_4*2) + ax1_0_16))
                    tir.reads([C_reindex_shared_dyn_warp_4[v0_o_24, v1_o_24, v2_o_24, 0:32, 0:8]])
                    tir.writes([C_reindex_shared_dyn_4[v0_o_24, (v1_o_24*16):((v1_o_24*16) + 16), (v2_o_24*16):((v2_o_24*16) + 16)]])
                    C_warp_18 = match_buffer(C_reindex_shared_dyn_warp_4[v0_o_24, v1_o_24, v2_o_24, 0:32, 0:8])
                    C_1_18 = match_buffer(C_reindex_shared_dyn_4[v0_o_24, (v1_o_24*16):((v1_o_24*16) + 16), (v2_o_24*16):((v2_o_24*16) + 16)])
                    for (tx_24: int32, 0, 32) "thread_binding" {
                      @tir.mma_store(16, 16, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), C_1_19: Pointer(shared.dyn float16), elem_offset_48: int32, (C_1_s0_4: int32*16), 2, dtype=handle), C_warp_19: Pointer(warp float16), elem_offset_49: int32, C_1_s0_4, dtype=float16)
                    }
                }
              }
            }
            for (ax0_ax1_ax2_fused_0_14: int32, 0, 8) "unroll" {
              for (ax0_ax1_ax2_fused_1_14: int32, 0, 32) "thread_binding" {
                for (ax0_ax1_ax2_fused_2_14: int32, 0, 8) "vectorized" {
                  block([1, 512, 128], "C_reindex_shared.dyn") as [v0_14, v1_14, v2_14] {
                    bind(v0_14, 0)
                    bind(v1_14, ((ax1_0_0_ax2_0_0_fused_4*16) + floordiv((((ax0_ax1_ax2_fused_0_14*256) + (ax0_ax1_ax2_fused_1_14*8)) + ax0_ax1_ax2_fused_2_14), 128)))
                    bind(v2_14, floormod((((ax0_ax1_ax2_fused_0_14*256) + (ax0_ax1_ax2_fused_1_14*8)) + ax0_ax1_ax2_fused_2_14), 128))
                    tir.reads([C_reindex_shared_dyn_4[v0_14, v1_14, v2_14]])
                    tir.writes([C_4[v1_14, v2_14]])
                    C_4[v1_14, v2_14] = C_reindex_shared_dyn_4[v0_14, v1_14, v2_14]
                }
              }
            }
          }
        }
      }
    }
}

@gemm_2 = primfn(A_handle_2: handle, B_handle_2: handle, C_handle_2: handle) -> ()
  attr = {"dlight.tensorcore_prenormlized": True, "global_symbol": "gemm_2", "op_pattern": 0}
  buffers = {A_6: Buffer(A_7: Pointer(global float16), float16, [512, 128], []),
             B_6: Buffer(B_7: Pointer(global float16), float16, [128, 128], []),
             C_6: Buffer(C_7: Pointer(global float16), float16, [512, 128], [])}
  buffer_map = {A_handle_2: A_6, B_handle_2: B_6, C_handle_2: C_6} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_reindex_shared_dyn_5 = alloc_buffer(float16[1, 512, 128])
    B_reindex_shared_dyn_5 = alloc_buffer(float16[1, 128, 128])
    A_reindex_shared_dyn_warp_5 = alloc_buffer(float16[1, 32, 8, 32, 8])
    B_reindex_shared_dyn_warp_5 = alloc_buffer(float16[1, 8, 8, 32, 8])
    C_reindex_shared_dyn_5 = alloc_buffer(float16[1, 512, 128])
    C_reindex_shared_dyn_warp_5 = alloc_buffer(float16[1, 32, 8, 32, 8])
    for (ax0_5: int32, 0, 1) "thread_binding" {
      for (ax1_0_0_ax2_0_0_fused_5: int32, 0, 32) "thread_binding" {
        for (ax1_0_1_ax2_0_1_fused_5: int32, 0, 1) "thread_binding" {
          for (ax1_0_2_5: int32, 0, 1) "thread_binding" {
            for (ax2_0_2_5: int32, 0, 4) "thread_binding" {
              for (ax1_0_3_init_5: int32, 0, 1) {
                for (ax2_0_3_init_5: int32, 0, 2) {
                  block([1, 32, 8], "gemm_o_init") as [v0_o_25, v1_o_25, v2_o_25] {
                    bind(v0_o_25, ax0_5)
                    bind(v1_o_25, ((ax1_0_0_ax2_0_0_fused_5 + ax1_0_2_5) + ax1_0_3_init_5))
                    bind(v2_o_25, ((ax2_0_2_5*2) + ax2_0_3_init_5))
                    tir.reads([])
                    tir.writes([C_reindex_shared_dyn_warp_5[0, v1_o_25, v2_o_25, 0:32, 0:8]])
                    block([1, 1], "gemm_init_o") as [v1_i_init_o_5, v2_i_init_o_5] {
                      bind(v1_i_init_o_5, 0)
                      bind(v2_i_init_o_5, 0)
                      tir.reads([])
                      tir.writes([C_reindex_shared_dyn_warp_5[0, v1_o_25, v2_o_25, 0:32, 0:8]])
                      C_warp_20 = match_buffer(C_reindex_shared_dyn_warp_5[0, v1_o_25, v2_o_25, 0:32, 0:8])
                      for (tx_25: int32, 0, 32) "thread_binding" {
                        @tir.mma_fill(8, C_warp_21: Pointer(warp float16), elem_offset_50: int32, dtype=float16)
                      }
                }
              }
              for (ax3_0_0_5: int32, 0, 1) {
                for (ax0_ax1_ax2_fused_0_15: int32, 0, 1) "thread_binding" {
                  for (ax0_ax1_ax2_fused_1_15: int32, 0, 4) "thread_binding" {
                    for (ax0_ax1_ax2_fused_2_15: int32, 0, 2) "unroll" {
                      for (ax0_ax1_ax2_fused_3_10: int32, 0, 32) "thread_binding" {
                        for (ax0_ax1_ax2_fused_4_10: int32, 0, 8) "vectorized" {
                          block([1, 512, 128], "A_reindex_shared.dyn") as [v0_15, v1_15, v2_15] {
                            bind(v0_15, 0)
                            bind(v1_15, ((ax1_0_0_ax2_0_0_fused_5*16) + floordiv((((((ax0_ax1_ax2_fused_0_15*2048) + (ax0_ax1_ax2_fused_1_15*512)) + (ax0_ax1_ax2_fused_2_15*256)) + (ax0_ax1_ax2_fused_3_10*8)) + ax0_ax1_ax2_fused_4_10), 128)))
                            bind(v2_15, floormod((((((ax0_ax1_ax2_fused_0_15*2048) + (ax0_ax1_ax2_fused_1_15*512)) + (ax0_ax1_ax2_fused_2_15*256)) + (ax0_ax1_ax2_fused_3_10*8)) + ax0_ax1_ax2_fused_4_10), 128))
                            tir.reads([A_6[v1_15, v2_15]])
                            tir.writes([A_reindex_shared_dyn_5[v0_15, v1_15, v2_15]])
                            tir.attrs({"permuted_layout": 0, "buffer_dim_align": [[0, 1, 16, 8]]})
                            A_reindex_shared_dyn_5[v0_15, v1_15, v2_15] = A_6[v1_15, v2_15]
                        }
                      }
                    }
                  }
                }
                for (ax0_ax1_ax2_fused_0_16: int32, 0, 1) "thread_binding" {
                  for (ax0_ax1_ax2_fused_1_16: int32, 0, 4) "thread_binding" {
                    for (ax0_ax1_ax2_fused_2_16: int32, 0, 16) "unroll" {
                      for (ax0_ax1_ax2_fused_3_11: int32, 0, 32) "thread_binding" {
                        for (ax0_ax1_ax2_fused_4_11: int32, 0, 8) "vectorized" {
                          block([1, 128, 128], "B_reindex_shared.dyn") as [v0_16, v1_16, v2_16] {
                            bind(v0_16, 0)
                            bind(v1_16, floordiv((((((ax0_ax1_ax2_fused_0_16*16384) + (ax0_ax1_ax2_fused_1_16*4096)) + (ax0_ax1_ax2_fused_2_16*256)) + (ax0_ax1_ax2_fused_3_11*8)) + ax0_ax1_ax2_fused_4_11), 128))
                            bind(v2_16, floormod((((((ax0_ax1_ax2_fused_0_16*16384) + (ax0_ax1_ax2_fused_1_16*4096)) + (ax0_ax1_ax2_fused_2_16*256)) + (ax0_ax1_ax2_fused_3_11*8)) + ax0_ax1_ax2_fused_4_11), 128))
                            tir.reads([B_6[v1_16, v2_16]])
                            tir.writes([B_reindex_shared_dyn_5[v0_16, v1_16, v2_16]])
                            tir.attrs({"permuted_layout": 0, "buffer_dim_align": [[0, 1, 16, 8]]})
                            B_reindex_shared_dyn_5[v0_16, v1_16, v2_16] = B_6[v1_16, v2_16]
                        }
                      }
                    }
                  }
                }
                for (ax3_0_1_5: int32, 0, 8) {
                  for (ax0_0_15: int32, 0, 1) {
                    for (ax1_0_17: int32, 0, 1) {
                      block([1, 32, 8], "A_reindex_shared.dyn_warp_o") as [v0_o_26, v1_o_26, v2_o_26] {
                        bind(v0_o_26, 0)
                        bind(v1_o_26, (ax1_0_0_ax2_0_0_fused_5 + ax0_0_15))
                        bind(v2_o_26, (ax3_0_1_5 + ax1_0_17))
                        tir.reads([A_reindex_shared_dyn_5[v0_o_26, (v1_o_26*16):((v1_o_26*16) + 16), (v2_o_26*16):((v2_o_26*16) + 16)]])
                        tir.writes([A_reindex_shared_dyn_warp_5[v0_o_26, v1_o_26, v2_o_26, 0:32, 0:8]])
                        tir.attrs({"permuted_layout": 0})
                        warp_20 = match_buffer(A_reindex_shared_dyn_warp_5[v0_o_26, v1_o_26, v2_o_26, 0:32, 0:8])
                        shared_20 = match_buffer(A_reindex_shared_dyn_5[v0_o_26, (v1_o_26*16):((v1_o_26*16) + 16), (v2_o_26*16):((v2_o_26*16) + 16)])
                        for (tx_26: int32, 0, 32) "thread_binding" {
                          @tir.ptx_ldmatrix(False, 4, ".b16", warp_21: Pointer(warp float16), (elem_offset_51: int32 + (8*tx_26)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_21: Pointer(shared.dyn float16), elem_offset_52: int32, (shared_s0_10: int32*16), 1, dtype=handle), ((shared_s0_10*floormod(tx_26, 16)) + (8*floordiv(tx_26, 16))), dtype=float16)
                        }
                    }
                  }
                  for (ax0_0_16: int32, 0, 1) {
                    for (ax1_0_18: int32, 0, 2) {
                      block([1, 8, 8], "B_reindex_shared.dyn_warp_o") as [v0_o_27, v1_o_27, v2_o_27] {
                        bind(v0_o_27, 0)
                        bind(v1_o_27, (ax3_0_1_5 + ax0_0_16))
                        bind(v2_o_27, ((ax2_0_2_5*2) + ax1_0_18))
                        tir.reads([B_reindex_shared_dyn_5[v0_o_27, (v1_o_27*16):((v1_o_27*16) + 16), (v2_o_27*16):((v2_o_27*16) + 16)]])
                        tir.writes([B_reindex_shared_dyn_warp_5[v0_o_27, v1_o_27, v2_o_27, 0:32, 0:8]])
                        tir.attrs({"permuted_layout": 0})
                        warp_22 = match_buffer(B_reindex_shared_dyn_warp_5[v0_o_27, v1_o_27, v2_o_27, 0:32, 0:8])
                        shared_22 = match_buffer(B_reindex_shared_dyn_5[v0_o_27, (v1_o_27*16):((v1_o_27*16) + 16), (v2_o_27*16):((v2_o_27*16) + 16)])
                        for (tx_27: int32, 0, 32) "thread_binding" {
                          @tir.ptx_ldmatrix(True, 4, ".b16", warp_23: Pointer(warp float16), (elem_offset_53: int32 + (8*tx_27)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_23: Pointer(shared.dyn float16), elem_offset_54: int32, (shared_s0_11: int32*16), 1, dtype=handle), ((shared_s0_11*floormod(tx_27, 16)) + (8*floordiv(tx_27, 16))), dtype=float16)
                        }
                    }
                  }
                  for (ax1_0_3_5: int32, 0, 1) {
                    for (ax2_0_3_5: int32, 0, 2) {
                      block([1, 32, 8, tir.reduce_axis(0, 8)], "gemm_o_update") as [v0_o_28, v1_o_28, v2_o_28, v3_o_5] {
                        bind(v0_o_28, ax0_5)
                        bind(v1_o_28, ((ax1_0_0_ax2_0_0_fused_5 + ax1_0_2_5) + ax1_0_3_5))
                        bind(v2_o_28, ((ax2_0_2_5*2) + ax2_0_3_5))
                        bind(v3_o_5, ((ax3_0_0_5*8) + ax3_0_1_5))
                        tir.reads([C_reindex_shared_dyn_warp_5[0, v1_o_28, v2_o_28, 0:32, 0:8], A_reindex_shared_dyn_warp_5[0, v1_o_28, v3_o_5, 0:32, 0:8], B_reindex_shared_dyn_warp_5[0, v3_o_5, v2_o_28, 0:32, 0:8]])
                        tir.writes([C_reindex_shared_dyn_warp_5[0, v1_o_28, v2_o_28, 0:32, 0:8]])
                        block([1, 1, tir.reduce_axis(0, 1)], "gemm_o") as [v1_i_o_5, v2_i_o_5, v3_i_o_5] {
                          bind(v1_i_o_5, 0)
                          bind(v2_i_o_5, 0)
                          bind(v3_i_o_5, 0)
                          tir.reads([C_reindex_shared_dyn_warp_5[0, v1_o_28, v2_o_28, 0:32, 0:8], A_reindex_shared_dyn_warp_5[0, v1_o_28, v3_o_5, 0:32, 0:8], B_reindex_shared_dyn_warp_5[0, v3_o_5, v2_o_28, 0:32, 0:8]])
                          tir.writes([C_reindex_shared_dyn_warp_5[0, v1_o_28, v2_o_28, 0:32, 0:8]])
                          A_1_11 = match_buffer(A_reindex_shared_dyn_warp_5[0, v1_o_28, v3_o_5, 0:32, 0:8])
                          B_1_11 = match_buffer(B_reindex_shared_dyn_warp_5[0, v3_o_5, v2_o_28, 0:32, 0:8])
                          C_1_20 = match_buffer(C_reindex_shared_dyn_warp_5[0, v1_o_28, v2_o_28, 0:32, 0:8])
                          for (tx_28: int32, 0, 32) "thread_binding" {
                            @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1_12: Pointer(warp float16), (elem_offset_55: int32 + (tx_28*8)), B_1_12: Pointer(warp float16), (elem_offset_56: int32 + (tx_28*8)), C_1_21: Pointer(warp float16), (elem_offset_57: int32 + (tx_28*8)), False, dtype=float16)
                            @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1_12, (elem_offset_55 + (tx_28*8)), B_1_12, ((elem_offset_56 + (tx_28*8)) + 4), C_1_21, ((elem_offset_57 + (tx_28*8)) + 4), False, dtype=float16)
                          }
                    }
                  }
                }
              }
              for (ax0_0_17: int32, 0, 1) {
                for (ax1_0_19: int32, 0, 2) {
                  block([1, 32, 8], "C_reindex_shared.dyn_warp_o") as [v0_o_29, v1_o_29, v2_o_29] {
                    bind(v0_o_29, 0)
                    bind(v1_o_29, ax1_0_0_ax2_0_0_fused_5)
                    bind(v2_o_29, ((ax2_0_2_5*2) + ax1_0_19))
                    tir.reads([C_reindex_shared_dyn_warp_5[v0_o_29, v1_o_29, v2_o_29, 0:32, 0:8]])
                    tir.writes([C_reindex_shared_dyn_5[v0_o_29, (v1_o_29*16):((v1_o_29*16) + 16), (v2_o_29*16):((v2_o_29*16) + 16)]])
                    C_warp_22 = match_buffer(C_reindex_shared_dyn_warp_5[v0_o_29, v1_o_29, v2_o_29, 0:32, 0:8])
                    C_1_22 = match_buffer(C_reindex_shared_dyn_5[v0_o_29, (v1_o_29*16):((v1_o_29*16) + 16), (v2_o_29*16):((v2_o_29*16) + 16)])
                    for (tx_29: int32, 0, 32) "thread_binding" {
                      @tir.mma_store(16, 16, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), C_1_23: Pointer(shared.dyn float16), elem_offset_58: int32, (C_1_s0_5: int32*16), 2, dtype=handle), C_warp_23: Pointer(warp float16), elem_offset_59: int32, C_1_s0_5, dtype=float16)
                    }
                }
              }
            }
            for (ax0_ax1_ax2_fused_0_17: int32, 0, 8) "unroll" {
              for (ax0_ax1_ax2_fused_1_17: int32, 0, 32) "thread_binding" {
                for (ax0_ax1_ax2_fused_2_17: int32, 0, 8) "vectorized" {
                  block([1, 512, 128], "C_reindex_shared.dyn") as [v0_17, v1_17, v2_17] {
                    bind(v0_17, 0)
                    bind(v1_17, ((ax1_0_0_ax2_0_0_fused_5*16) + floordiv((((ax0_ax1_ax2_fused_0_17*256) + (ax0_ax1_ax2_fused_1_17*8)) + ax0_ax1_ax2_fused_2_17), 128)))
                    bind(v2_17, floormod((((ax0_ax1_ax2_fused_0_17*256) + (ax0_ax1_ax2_fused_1_17*8)) + ax0_ax1_ax2_fused_2_17), 128))
                    tir.reads([C_reindex_shared_dyn_5[v0_17, v1_17, v2_17]])
                    tir.writes([C_6[v1_17, v2_17]])
                    C_6[v1_17, v2_17] = C_reindex_shared_dyn_5[v0_17, v1_17, v2_17]
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