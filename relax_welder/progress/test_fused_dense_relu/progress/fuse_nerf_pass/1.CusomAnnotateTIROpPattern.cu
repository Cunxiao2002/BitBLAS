#[version = "0.0.5"]
@fused_dense_relu_0 = primfn(input0_handle: handle, param_0_handle: handle, T_relu_intermediate_handle: handle) -> ()
  attr = {"tir.noalias": True, "dlight.tensorcore_prenormlized": True, "global_symbol": "fused_dense_relu_0", "op_pattern": 0}
  buffers = {input0: Buffer(input0_1: Pointer(global float16), float16, [2073600i64, 64i64], []),
             param_0: Buffer(param_0_1: Pointer(global float16), float16, [64i64, 64i64], []),
             T_relu_intermediate: Buffer(T_relu_intermediate_1: Pointer(global float16), float16, [2073600i64, 64i64], [])}
  buffer_map = {input0_handle: input0, param_0_handle: param_0, T_relu_intermediate_handle: T_relu_intermediate} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    input0_reindex_shared_dyn = alloc_buffer(float16[1i64, 2073600i64, 64i64])
    param_0_reindex_shared_dyn = alloc_buffer(float16[1i64, 64i64, 64i64])
    input0_reindex_shared_dyn_warp = alloc_buffer(float16[1i64, 129600i64, 4i64, 32i64, 8i64])
    param_0_reindex_shared_dyn_warp = alloc_buffer(float16[1i64, 4i64, 4i64, 32i64, 8i64])
    T_matmul_NT_intermediate_reindex_shared_dyn = alloc_buffer(float16[1i64, 2073600i64, 64i64])
    T_matmul_NT_intermediate_reindex_shared_dyn_warp = alloc_buffer(float16[1i64, 129600i64, 4i64, 32i64, 8i64])
    for (ax0: int64, 0i64, 1i64) "thread_binding" {
      for (ax1_0_0_ax2_0_0_fused: int64, 0i64, 21600i64) "thread_binding" {
        for (ax1_0_1_ax2_0_1_fused: int64, 0i64, 1i64) "thread_binding" {
          for (ax1_0_2: int64, 0i64, 2i64) "thread_binding" {
            for (ax2_0_2: int64, 0i64, 2i64) "thread_binding" {
              for (ax1_0_3_init: int64, 0i64, 3i64) {
                for (ax2_0_3_init: int64, 0i64, 2i64) {
                  block([1i64, 129600i64, 4i64], "T_matmul_NT_o_init") as [v0_o, v1_o, v2_o] {
                    bind(v0_o, ax0)
                    bind(v1_o, (((ax1_0_0_ax2_0_0_fused*6i64) + (ax1_0_2*3i64)) + ax1_0_3_init))
                    bind(v2_o, ((ax2_0_2*2i64) + ax2_0_3_init))
                    tir.reads([])
                    tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn_warp[0i64, v1_o, v2_o, 0i64:32i64, 0i64:8i64]])
                    block([1i64, 1i64], "T_matmul_NT_init_o") as [v1_i_init_o, v2_i_init_o] {
                      bind(v1_i_init_o, 0i64)
                      bind(v2_i_init_o, 0i64)
                      tir.reads([])
                      tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn_warp[0i64, v1_o, v2_o, 0i64:32i64, 0i64:8i64]])
                      C_warp = match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_warp[0i64, v1_o, v2_o, 0i64:32i64, 0i64:8i64])
                      for (tx: int64, 0i64, 32i64) "thread_binding" {
                        @tir.mma_fill(8, C_warp_1: Pointer(warp float16), elem_offset: int64, dtype=float16)
                      }
                }
              }
              for (ax3_0_0: int64, 0i64, 2i64) {
                for (ax0_ax1_ax2_fused_0: int64, 0i64, 2i64) "thread_binding" {
                  for (ax0_ax1_ax2_fused_1: int64, 0i64, 2i64) "thread_binding" {
                    for (ax0_ax1_ax2_fused_2: int64, 0i64, 3i64) "unroll" {
                      for (ax0_ax1_ax2_fused_3: int64, 0i64, 32i64) "thread_binding" {
                        for (ax0_ax1_ax2_fused_4: int64, 0i64, 8i64) "vectorized" {
                          block([1i64, 2073600i64, 64i64], "input0_reindex_shared.dyn") as [v0, v1, v2] {
                            bind(v0, 0i64)
                            bind(v1, ((ax1_0_0_ax2_0_0_fused*96i64) + floordiv((((((ax0_ax1_ax2_fused_0*1536i64) + (ax0_ax1_ax2_fused_1*768i64)) + (ax0_ax1_ax2_fused_2*256i64)) + (ax0_ax1_ax2_fused_3*8i64)) + ax0_ax1_ax2_fused_4), 32i64)))
                            bind(v2, ((ax3_0_0*32i64) + floormod((((((ax0_ax1_ax2_fused_0*1536i64) + (ax0_ax1_ax2_fused_1*768i64)) + (ax0_ax1_ax2_fused_2*256i64)) + (ax0_ax1_ax2_fused_3*8i64)) + ax0_ax1_ax2_fused_4), 32i64)))
                            tir.reads([input0[v1, v2]])
                            tir.writes([input0_reindex_shared_dyn[v0, v1, v2]])
                            tir.attrs({"permuted_layout": 1})
                            input0_reindex_shared_dyn[v0, v1, v2] = input0[v1, v2]
                        }
                      }
                    }
                  }
                }
                for (ax0_ax1_ax2_fused_0_1: int64, 0i64, 2i64) "thread_binding" {
                  for (ax0_ax1_ax2_fused_1_1: int64, 0i64, 2i64) "thread_binding" {
                    for (ax0_ax1_ax2_fused_2_1: int64, 0i64, 2i64) "unroll" {
                      for (ax0_ax1_ax2_fused_3_1: int64, 0i64, 32i64) "thread_binding" {
                        for (ax0_ax1_ax2_fused_4_1: int64, 0i64, 8i64) "vectorized" {
                          block([1i64, 64i64, 64i64], "param_0_reindex_shared.dyn") as [v0_1, v1_1, v2_1] {
                            bind(v0_1, 0i64)
                            bind(v1_1, floordiv((((((ax0_ax1_ax2_fused_0_1*1024i64) + (ax0_ax1_ax2_fused_1_1*512i64)) + (ax0_ax1_ax2_fused_2_1*256i64)) + (ax0_ax1_ax2_fused_3_1*8i64)) + ax0_ax1_ax2_fused_4_1), 32i64))
                            bind(v2_1, ((ax3_0_0*32i64) + floormod((((((ax0_ax1_ax2_fused_0_1*1024i64) + (ax0_ax1_ax2_fused_1_1*512i64)) + (ax0_ax1_ax2_fused_2_1*256i64)) + (ax0_ax1_ax2_fused_3_1*8i64)) + ax0_ax1_ax2_fused_4_1), 32i64)))
                            tir.reads([param_0[v1_1, v2_1]])
                            tir.writes([param_0_reindex_shared_dyn[v0_1, v1_1, v2_1]])
                            tir.attrs({"permuted_layout": 1})
                            param_0_reindex_shared_dyn[v0_1, v1_1, v2_1] = param_0[v1_1, v2_1]
                        }
                      }
                    }
                  }
                }
                for (ax3_0_1: int64, 0i64, 2i64) {
                  for (ax0_0: int64, 0i64, 3i64) {
                    for (ax1_0: int64, 0i64, 1i64) {
                      block([1i64, 129600i64, 4i64], "input0_reindex_shared.dyn_warp_o") as [v0_o_1, v1_o_1, v2_o_1] {
                        bind(v0_o_1, 0i64)
                        bind(v1_o_1, (((ax1_0_0_ax2_0_0_fused*6i64) + (ax1_0_2*3i64)) + ax0_0))
                        bind(v2_o_1, (((ax3_0_0*2i64) + ax3_0_1) + ax1_0))
                        tir.reads([input0_reindex_shared_dyn[v0_o_1, (v1_o_1*16i64):((v1_o_1*16i64) + 16i64), (v2_o_1*16i64):((v2_o_1*16i64) + 16i64)]])
                        tir.writes([input0_reindex_shared_dyn_warp[v0_o_1, v1_o_1, v2_o_1, 0i64:32i64, 0i64:8i64]])
                        tir.attrs({"permuted_layout": 1})
                        warp = match_buffer(input0_reindex_shared_dyn_warp[v0_o_1, v1_o_1, v2_o_1, 0i64:32i64, 0i64:8i64])
                        shared = match_buffer(input0_reindex_shared_dyn[v0_o_1, (v1_o_1*16i64):((v1_o_1*16i64) + 16i64), (v2_o_1*16i64):((v2_o_1*16i64) + 16i64)])
                        for (tx_1: int64, 0i64, 32i64) "thread_binding" {
                          @tir.ptx_ldmatrix(False, 4, ".b16", warp_1: Pointer(warp float16), (elem_offset_1: int64 + (8i64*tx_1)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_1: Pointer(shared.dyn float16), elem_offset_2: int64, (shared_s0: int64*16i64), 1, dtype=handle), ((shared_s0*floormod(tx_1, 16i64)) + (8i64*floordiv(tx_1, 16i64))), dtype=float16)
                        }
                    }
                  }
                  for (ax0_0_1: int64, 0i64, 2i64) {
                    for (ax1_0_1: int64, 0i64, 1i64) {
                      block([1i64, 4i64, 4i64], "param_0_reindex_shared.dyn_warp_o") as [v0_o_2, v1_o_2, v2_o_2] {
                        bind(v0_o_2, 0i64)
                        bind(v1_o_2, ((ax2_0_2*2i64) + ax0_0_1))
                        bind(v2_o_2, (((ax3_0_0*2i64) + ax3_0_1) + ax1_0_1))
                        tir.reads([param_0_reindex_shared_dyn[v0_o_2, (v1_o_2*16i64):((v1_o_2*16i64) + 16i64), (v2_o_2*16i64):((v2_o_2*16i64) + 16i64)]])
                        tir.writes([param_0_reindex_shared_dyn_warp[v0_o_2, v1_o_2, v2_o_2, 0i64:32i64, 0i64:8i64]])
                        tir.attrs({"permuted_layout": 1})
                        warp_2 = match_buffer(param_0_reindex_shared_dyn_warp[v0_o_2, v1_o_2, v2_o_2, 0i64:32i64, 0i64:8i64])
                        shared_2 = match_buffer(param_0_reindex_shared_dyn[v0_o_2, (v1_o_2*16i64):((v1_o_2*16i64) + 16i64), (v2_o_2*16i64):((v2_o_2*16i64) + 16i64)])
                        for (tx_2: int64, 0i64, 32i64) "thread_binding" {
                          @tir.ptx_ldmatrix(False, 4, ".b16", warp_3: Pointer(warp float16), (elem_offset_3: int64 + (8i64*tx_2)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_3: Pointer(shared.dyn float16), elem_offset_4: int64, (shared_s0_1: int64*16i64), 1, dtype=handle), ((((shared_s0_1*8i64)*floordiv(tx_2, 16i64)) + (shared_s0_1*floormod(tx_2, 8i64))) + (8i64*floordiv(floormod(tx_2, 16i64), 8i64))), dtype=float16)
                        }
                    }
                  }
                  for (ax1_0_3: int64, 0i64, 3i64) {
                    for (ax2_0_3: int64, 0i64, 2i64) {
                      block([1i64, 129600i64, 4i64, tir.reduce_axis(0i64, 4i64)], "T_matmul_NT_o_update") as [v0_o_3, v1_o_3, v2_o_3, v3_o] {
                        bind(v0_o_3, ax0)
                        bind(v1_o_3, (((ax1_0_0_ax2_0_0_fused*6i64) + (ax1_0_2*3i64)) + ax1_0_3))
                        bind(v2_o_3, ((ax2_0_2*2i64) + ax2_0_3))
                        bind(v3_o, ((ax3_0_0*2i64) + ax3_0_1))
                        tir.reads([T_matmul_NT_intermediate_reindex_shared_dyn_warp[0i64, v1_o_3, v2_o_3, 0i64:32i64, 0i64:8i64], input0_reindex_shared_dyn_warp[0i64, v1_o_3, v3_o, 0i64:32i64, 0i64:8i64], param_0_reindex_shared_dyn_warp[0i64, v2_o_3, v3_o, 0i64:32i64, 0i64:8i64]])
                        tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn_warp[0i64, v1_o_3, v2_o_3, 0i64:32i64, 0i64:8i64]])
                        block([1i64, 1i64, tir.reduce_axis(0i64, 1i64)], "T_matmul_NT_o") as [v1_i_o, v2_i_o, v3_i_o] {
                          bind(v1_i_o, 0i64)
                          bind(v2_i_o, 0i64)
                          bind(v3_i_o, 0i64)
                          tir.reads([T_matmul_NT_intermediate_reindex_shared_dyn_warp[0i64, v1_o_3, v2_o_3, 0i64:32i64, 0i64:8i64], input0_reindex_shared_dyn_warp[0i64, v1_o_3, v3_o, 0i64:32i64, 0i64:8i64], param_0_reindex_shared_dyn_warp[0i64, v2_o_3, v3_o, 0i64:32i64, 0i64:8i64]])
                          tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn_warp[0i64, v1_o_3, v2_o_3, 0i64:32i64, 0i64:8i64]])
                          A = match_buffer(input0_reindex_shared_dyn_warp[0i64, v1_o_3, v3_o, 0i64:32i64, 0i64:8i64])
                          B = match_buffer(param_0_reindex_shared_dyn_warp[0i64, v2_o_3, v3_o, 0i64:32i64, 0i64:8i64])
                          C = match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_warp[0i64, v1_o_3, v2_o_3, 0i64:32i64, 0i64:8i64])
                          for (tx_3: int64, 0i64, 32i64) "thread_binding" {
                            @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1: Pointer(warp float16), (elem_offset_5: int64 + (tx_3*8i64)), B_1: Pointer(warp float16), (elem_offset_6: int64 + (tx_3*8i64)), C_1: Pointer(warp float16), (elem_offset_7: int64 + (tx_3*8i64)), False, dtype=float16)
                            @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1, (elem_offset_5 + (tx_3*8i64)), B_1, ((elem_offset_6 + (tx_3*8i64)) + 4i64), C_1, ((elem_offset_7 + (tx_3*8i64)) + 4i64), False, dtype=float16)
                          }
                    }
                  }
                }
              }
              for (ax0_0_2: int64, 0i64, 3i64) {
                for (ax1_0_4: int64, 0i64, 2i64) {
                  block([1i64, 129600i64, 4i64], "T_matmul_NT_intermediate_reindex_shared.dyn_warp_o") as [v0_o_4, v1_o_4, v2_o_4] {
                    bind(v0_o_4, 0i64)
                    bind(v1_o_4, (((ax1_0_0_ax2_0_0_fused*6i64) + (ax1_0_2*3i64)) + ax0_0_2))
                    bind(v2_o_4, ((ax2_0_2*2i64) + ax1_0_4))
                    tir.reads([T_matmul_NT_intermediate_reindex_shared_dyn_warp[v0_o_4, v1_o_4, v2_o_4, 0i64:32i64, 0i64:8i64]])
                    tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn[v0_o_4, (v1_o_4*16i64):((v1_o_4*16i64) + 16i64), (v2_o_4*16i64):((v2_o_4*16i64) + 16i64)]])
                    C_warp_2 = match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_warp[v0_o_4, v1_o_4, v2_o_4, 0i64:32i64, 0i64:8i64])
                    C_2 = match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn[v0_o_4, (v1_o_4*16i64):((v1_o_4*16i64) + 16i64), (v2_o_4*16i64):((v2_o_4*16i64) + 16i64)])
                    for (tx_4: int64, 0i64, 32i64) "thread_binding" {
                      @tir.mma_store(16, 16, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), C_3: Pointer(shared.dyn float16), elem_offset_8: int64, (C_s0: int64*16i64), 2, dtype=handle), C_warp_3: Pointer(warp float16), elem_offset_9: int64, C_s0, dtype=float16)
                    }
                }
              }
            }
            for (ax0_ax1_ax2_fused_0_2: int64, 0i64, 12i64) "unroll" {
              for (ax0_ax1_ax2_fused_1_2: int64, 0i64, 32i64) "thread_binding" {
                for (ax0_ax1_ax2_fused_2_2: int64, 0i64, 8i64) "vectorized" {
                  block([1i64, 2073600i64, 64i64], "T_matmul_NT_intermediate_reindex_shared.dyn") as [v0_2, v1_2, v2_2] {
                    bind(v0_2, 0i64)
                    bind(v1_2, (((ax1_0_0_ax2_0_0_fused*96i64) + (ax1_0_2*48i64)) + floordiv((((ax0_ax1_ax2_fused_0_2*256i64) + (ax0_ax1_ax2_fused_1_2*8i64)) + ax0_ax1_ax2_fused_2_2), 64i64)))
                    bind(v2_2, floormod((((ax0_ax1_ax2_fused_0_2*256i64) + (ax0_ax1_ax2_fused_1_2*8i64)) + ax0_ax1_ax2_fused_2_2), 64i64))
                    tir.reads([T_matmul_NT_intermediate_reindex_shared_dyn[v0_2, v1_2, v2_2]])
                    tir.writes([T_relu_intermediate[v1_2, v2_2]])
                    T_relu_intermediate[v1_2, v2_2] = max(T_matmul_NT_intermediate_reindex_shared_dyn[v0_2, v1_2, v2_2], 0f16)
                }
              }
            }
          }
        }
      }
    }
}

@fused_dense_relu_1 = primfn(input0_handle_1: handle, param_0_handle_1: handle, T_relu_intermediate_handle_1: handle) -> ()
  attr = {"tir.noalias": True, "dlight.tensorcore_prenormlized": True, "global_symbol": "fused_dense_relu_1", "op_pattern": 0}
  buffers = {input0_2: Buffer(input0_3: Pointer(global float16), float16, [2073600i64, 64i64], []),
             param_0_2: Buffer(param_0_3: Pointer(global float16), float16, [64i64, 64i64], []),
             T_relu_intermediate_2: Buffer(T_relu_intermediate_3: Pointer(global float16), float16, [2073600i64, 64i64], [])}
  buffer_map = {input0_handle_1: input0_2, param_0_handle_1: param_0_2, T_relu_intermediate_handle_1: T_relu_intermediate_2} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    input0_reindex_shared_dyn_1 = alloc_buffer(float16[1i64, 2073600i64, 64i64])
    param_0_reindex_shared_dyn_1 = alloc_buffer(float16[1i64, 64i64, 64i64])
    input0_reindex_shared_dyn_warp_1 = alloc_buffer(float16[1i64, 129600i64, 4i64, 32i64, 8i64])
    param_0_reindex_shared_dyn_warp_1 = alloc_buffer(float16[1i64, 4i64, 4i64, 32i64, 8i64])
    T_matmul_NT_intermediate_reindex_shared_dyn_1 = alloc_buffer(float16[1i64, 2073600i64, 64i64])
    T_matmul_NT_intermediate_reindex_shared_dyn_warp_1 = alloc_buffer(float16[1i64, 129600i64, 4i64, 32i64, 8i64])
    for (ax0_1: int64, 0i64, 1i64) "thread_binding" {
      for (ax1_0_0_ax2_0_0_fused_1: int64, 0i64, 21600i64) "thread_binding" {
        for (ax1_0_1_ax2_0_1_fused_1: int64, 0i64, 1i64) "thread_binding" {
          for (ax1_0_2_1: int64, 0i64, 2i64) "thread_binding" {
            for (ax2_0_2_1: int64, 0i64, 2i64) "thread_binding" {
              for (ax1_0_3_init_1: int64, 0i64, 3i64) {
                for (ax2_0_3_init_1: int64, 0i64, 2i64) {
                  block([1i64, 129600i64, 4i64], "T_matmul_NT_o_init") as [v0_o_5, v1_o_5, v2_o_5] {
                    bind(v0_o_5, ax0_1)
                    bind(v1_o_5, (((ax1_0_0_ax2_0_0_fused_1*6i64) + (ax1_0_2_1*3i64)) + ax1_0_3_init_1))
                    bind(v2_o_5, ((ax2_0_2_1*2i64) + ax2_0_3_init_1))
                    tir.reads([])
                    tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[0i64, v1_o_5, v2_o_5, 0i64:32i64, 0i64:8i64]])
                    block([1i64, 1i64], "T_matmul_NT_init_o") as [v1_i_init_o_1, v2_i_init_o_1] {
                      bind(v1_i_init_o_1, 0i64)
                      bind(v2_i_init_o_1, 0i64)
                      tir.reads([])
                      tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[0i64, v1_o_5, v2_o_5, 0i64:32i64, 0i64:8i64]])
                      C_warp_4 = match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[0i64, v1_o_5, v2_o_5, 0i64:32i64, 0i64:8i64])
                      for (tx_5: int64, 0i64, 32i64) "thread_binding" {
                        @tir.mma_fill(8, C_warp_5: Pointer(warp float16), elem_offset_10: int64, dtype=float16)
                      }
                }
              }
              for (ax3_0_0_1: int64, 0i64, 2i64) {
                for (ax0_ax1_ax2_fused_0_3: int64, 0i64, 2i64) "thread_binding" {
                  for (ax0_ax1_ax2_fused_1_3: int64, 0i64, 2i64) "thread_binding" {
                    for (ax0_ax1_ax2_fused_2_3: int64, 0i64, 3i64) "unroll" {
                      for (ax0_ax1_ax2_fused_3_2: int64, 0i64, 32i64) "thread_binding" {
                        for (ax0_ax1_ax2_fused_4_2: int64, 0i64, 8i64) "vectorized" {
                          block([1i64, 2073600i64, 64i64], "input0_reindex_shared.dyn") as [v0_3, v1_3, v2_3] {
                            bind(v0_3, 0i64)
                            bind(v1_3, ((ax1_0_0_ax2_0_0_fused_1*96i64) + floordiv((((((ax0_ax1_ax2_fused_0_3*1536i64) + (ax0_ax1_ax2_fused_1_3*768i64)) + (ax0_ax1_ax2_fused_2_3*256i64)) + (ax0_ax1_ax2_fused_3_2*8i64)) + ax0_ax1_ax2_fused_4_2), 32i64)))
                            bind(v2_3, ((ax3_0_0_1*32i64) + floormod((((((ax0_ax1_ax2_fused_0_3*1536i64) + (ax0_ax1_ax2_fused_1_3*768i64)) + (ax0_ax1_ax2_fused_2_3*256i64)) + (ax0_ax1_ax2_fused_3_2*8i64)) + ax0_ax1_ax2_fused_4_2), 32i64)))
                            tir.reads([input0_2[v1_3, v2_3]])
                            tir.writes([input0_reindex_shared_dyn_1[v0_3, v1_3, v2_3]])
                            tir.attrs({"permuted_layout": 1})
                            input0_reindex_shared_dyn_1[v0_3, v1_3, v2_3] = input0_2[v1_3, v2_3]
                        }
                      }
                    }
                  }
                }
                for (ax0_ax1_ax2_fused_0_4: int64, 0i64, 2i64) "thread_binding" {
                  for (ax0_ax1_ax2_fused_1_4: int64, 0i64, 2i64) "thread_binding" {
                    for (ax0_ax1_ax2_fused_2_4: int64, 0i64, 2i64) "unroll" {
                      for (ax0_ax1_ax2_fused_3_3: int64, 0i64, 32i64) "thread_binding" {
                        for (ax0_ax1_ax2_fused_4_3: int64, 0i64, 8i64) "vectorized" {
                          block([1i64, 64i64, 64i64], "param_0_reindex_shared.dyn") as [v0_4, v1_4, v2_4] {
                            bind(v0_4, 0i64)
                            bind(v1_4, floordiv((((((ax0_ax1_ax2_fused_0_4*1024i64) + (ax0_ax1_ax2_fused_1_4*512i64)) + (ax0_ax1_ax2_fused_2_4*256i64)) + (ax0_ax1_ax2_fused_3_3*8i64)) + ax0_ax1_ax2_fused_4_3), 32i64))
                            bind(v2_4, ((ax3_0_0_1*32i64) + floormod((((((ax0_ax1_ax2_fused_0_4*1024i64) + (ax0_ax1_ax2_fused_1_4*512i64)) + (ax0_ax1_ax2_fused_2_4*256i64)) + (ax0_ax1_ax2_fused_3_3*8i64)) + ax0_ax1_ax2_fused_4_3), 32i64)))
                            tir.reads([param_0_2[v1_4, v2_4]])
                            tir.writes([param_0_reindex_shared_dyn_1[v0_4, v1_4, v2_4]])
                            tir.attrs({"permuted_layout": 1})
                            param_0_reindex_shared_dyn_1[v0_4, v1_4, v2_4] = param_0_2[v1_4, v2_4]
                        }
                      }
                    }
                  }
                }
                for (ax3_0_1_1: int64, 0i64, 2i64) {
                  for (ax0_0_3: int64, 0i64, 3i64) {
                    for (ax1_0_5: int64, 0i64, 1i64) {
                      block([1i64, 129600i64, 4i64], "input0_reindex_shared.dyn_warp_o") as [v0_o_6, v1_o_6, v2_o_6] {
                        bind(v0_o_6, 0i64)
                        bind(v1_o_6, (((ax1_0_0_ax2_0_0_fused_1*6i64) + (ax1_0_2_1*3i64)) + ax0_0_3))
                        bind(v2_o_6, (((ax3_0_0_1*2i64) + ax3_0_1_1) + ax1_0_5))
                        tir.reads([input0_reindex_shared_dyn_1[v0_o_6, (v1_o_6*16i64):((v1_o_6*16i64) + 16i64), (v2_o_6*16i64):((v2_o_6*16i64) + 16i64)]])
                        tir.writes([input0_reindex_shared_dyn_warp_1[v0_o_6, v1_o_6, v2_o_6, 0i64:32i64, 0i64:8i64]])
                        tir.attrs({"permuted_layout": 1})
                        warp_4 = match_buffer(input0_reindex_shared_dyn_warp_1[v0_o_6, v1_o_6, v2_o_6, 0i64:32i64, 0i64:8i64])
                        shared_4 = match_buffer(input0_reindex_shared_dyn_1[v0_o_6, (v1_o_6*16i64):((v1_o_6*16i64) + 16i64), (v2_o_6*16i64):((v2_o_6*16i64) + 16i64)])
                        for (tx_6: int64, 0i64, 32i64) "thread_binding" {
                          @tir.ptx_ldmatrix(False, 4, ".b16", warp_5: Pointer(warp float16), (elem_offset_11: int64 + (8i64*tx_6)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_5: Pointer(shared.dyn float16), elem_offset_12: int64, (shared_s0_2: int64*16i64), 1, dtype=handle), ((shared_s0_2*floormod(tx_6, 16i64)) + (8i64*floordiv(tx_6, 16i64))), dtype=float16)
                        }
                    }
                  }
                  for (ax0_0_4: int64, 0i64, 2i64) {
                    for (ax1_0_6: int64, 0i64, 1i64) {
                      block([1i64, 4i64, 4i64], "param_0_reindex_shared.dyn_warp_o") as [v0_o_7, v1_o_7, v2_o_7] {
                        bind(v0_o_7, 0i64)
                        bind(v1_o_7, ((ax2_0_2_1*2i64) + ax0_0_4))
                        bind(v2_o_7, (((ax3_0_0_1*2i64) + ax3_0_1_1) + ax1_0_6))
                        tir.reads([param_0_reindex_shared_dyn_1[v0_o_7, (v1_o_7*16i64):((v1_o_7*16i64) + 16i64), (v2_o_7*16i64):((v2_o_7*16i64) + 16i64)]])
                        tir.writes([param_0_reindex_shared_dyn_warp_1[v0_o_7, v1_o_7, v2_o_7, 0i64:32i64, 0i64:8i64]])
                        tir.attrs({"permuted_layout": 1})
                        warp_6 = match_buffer(param_0_reindex_shared_dyn_warp_1[v0_o_7, v1_o_7, v2_o_7, 0i64:32i64, 0i64:8i64])
                        shared_6 = match_buffer(param_0_reindex_shared_dyn_1[v0_o_7, (v1_o_7*16i64):((v1_o_7*16i64) + 16i64), (v2_o_7*16i64):((v2_o_7*16i64) + 16i64)])
                        for (tx_7: int64, 0i64, 32i64) "thread_binding" {
                          @tir.ptx_ldmatrix(False, 4, ".b16", warp_7: Pointer(warp float16), (elem_offset_13: int64 + (8i64*tx_7)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_7: Pointer(shared.dyn float16), elem_offset_14: int64, (shared_s0_3: int64*16i64), 1, dtype=handle), ((((shared_s0_3*8i64)*floordiv(tx_7, 16i64)) + (shared_s0_3*floormod(tx_7, 8i64))) + (8i64*floordiv(floormod(tx_7, 16i64), 8i64))), dtype=float16)
                        }
                    }
                  }
                  for (ax1_0_3_1: int64, 0i64, 3i64) {
                    for (ax2_0_3_1: int64, 0i64, 2i64) {
                      block([1i64, 129600i64, 4i64, tir.reduce_axis(0i64, 4i64)], "T_matmul_NT_o_update") as [v0_o_8, v1_o_8, v2_o_8, v3_o_1] {
                        bind(v0_o_8, ax0_1)
                        bind(v1_o_8, (((ax1_0_0_ax2_0_0_fused_1*6i64) + (ax1_0_2_1*3i64)) + ax1_0_3_1))
                        bind(v2_o_8, ((ax2_0_2_1*2i64) + ax2_0_3_1))
                        bind(v3_o_1, ((ax3_0_0_1*2i64) + ax3_0_1_1))
                        tir.reads([T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[0i64, v1_o_8, v2_o_8, 0i64:32i64, 0i64:8i64], input0_reindex_shared_dyn_warp_1[0i64, v1_o_8, v3_o_1, 0i64:32i64, 0i64:8i64], param_0_reindex_shared_dyn_warp_1[0i64, v2_o_8, v3_o_1, 0i64:32i64, 0i64:8i64]])
                        tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[0i64, v1_o_8, v2_o_8, 0i64:32i64, 0i64:8i64]])
                        block([1i64, 1i64, tir.reduce_axis(0i64, 1i64)], "T_matmul_NT_o") as [v1_i_o_1, v2_i_o_1, v3_i_o_1] {
                          bind(v1_i_o_1, 0i64)
                          bind(v2_i_o_1, 0i64)
                          bind(v3_i_o_1, 0i64)
                          tir.reads([T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[0i64, v1_o_8, v2_o_8, 0i64:32i64, 0i64:8i64], input0_reindex_shared_dyn_warp_1[0i64, v1_o_8, v3_o_1, 0i64:32i64, 0i64:8i64], param_0_reindex_shared_dyn_warp_1[0i64, v2_o_8, v3_o_1, 0i64:32i64, 0i64:8i64]])
                          tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[0i64, v1_o_8, v2_o_8, 0i64:32i64, 0i64:8i64]])
                          A_2 = match_buffer(input0_reindex_shared_dyn_warp_1[0i64, v1_o_8, v3_o_1, 0i64:32i64, 0i64:8i64])
                          B_2 = match_buffer(param_0_reindex_shared_dyn_warp_1[0i64, v2_o_8, v3_o_1, 0i64:32i64, 0i64:8i64])
                          C_4 = match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[0i64, v1_o_8, v2_o_8, 0i64:32i64, 0i64:8i64])
                          for (tx_8: int64, 0i64, 32i64) "thread_binding" {
                            @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_3: Pointer(warp float16), (elem_offset_15: int64 + (tx_8*8i64)), B_3: Pointer(warp float16), (elem_offset_16: int64 + (tx_8*8i64)), C_5: Pointer(warp float16), (elem_offset_17: int64 + (tx_8*8i64)), False, dtype=float16)
                            @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_3, (elem_offset_15 + (tx_8*8i64)), B_3, ((elem_offset_16 + (tx_8*8i64)) + 4i64), C_5, ((elem_offset_17 + (tx_8*8i64)) + 4i64), False, dtype=float16)
                          }
                    }
                  }
                }
              }
              for (ax0_0_5: int64, 0i64, 3i64) {
                for (ax1_0_7: int64, 0i64, 2i64) {
                  block([1i64, 129600i64, 4i64], "T_matmul_NT_intermediate_reindex_shared.dyn_warp_o") as [v0_o_9, v1_o_9, v2_o_9] {
                    bind(v0_o_9, 0i64)
                    bind(v1_o_9, (((ax1_0_0_ax2_0_0_fused_1*6i64) + (ax1_0_2_1*3i64)) + ax0_0_5))
                    bind(v2_o_9, ((ax2_0_2_1*2i64) + ax1_0_7))
                    tir.reads([T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[v0_o_9, v1_o_9, v2_o_9, 0i64:32i64, 0i64:8i64]])
                    tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn_1[v0_o_9, (v1_o_9*16i64):((v1_o_9*16i64) + 16i64), (v2_o_9*16i64):((v2_o_9*16i64) + 16i64)]])
                    C_warp_6 = match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[v0_o_9, v1_o_9, v2_o_9, 0i64:32i64, 0i64:8i64])
                    C_6 = match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_1[v0_o_9, (v1_o_9*16i64):((v1_o_9*16i64) + 16i64), (v2_o_9*16i64):((v2_o_9*16i64) + 16i64)])
                    for (tx_9: int64, 0i64, 32i64) "thread_binding" {
                      @tir.mma_store(16, 16, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), C_7: Pointer(shared.dyn float16), elem_offset_18: int64, (C_s0_1: int64*16i64), 2, dtype=handle), C_warp_7: Pointer(warp float16), elem_offset_19: int64, C_s0_1, dtype=float16)
                    }
                }
              }
            }
            for (ax0_ax1_ax2_fused_0_5: int64, 0i64, 12i64) "unroll" {
              for (ax0_ax1_ax2_fused_1_5: int64, 0i64, 32i64) "thread_binding" {
                for (ax0_ax1_ax2_fused_2_5: int64, 0i64, 8i64) "vectorized" {
                  block([1i64, 2073600i64, 64i64], "T_matmul_NT_intermediate_reindex_shared.dyn") as [v0_5, v1_5, v2_5] {
                    bind(v0_5, 0i64)
                    bind(v1_5, (((ax1_0_0_ax2_0_0_fused_1*96i64) + (ax1_0_2_1*48i64)) + floordiv((((ax0_ax1_ax2_fused_0_5*256i64) + (ax0_ax1_ax2_fused_1_5*8i64)) + ax0_ax1_ax2_fused_2_5), 64i64)))
                    bind(v2_5, floormod((((ax0_ax1_ax2_fused_0_5*256i64) + (ax0_ax1_ax2_fused_1_5*8i64)) + ax0_ax1_ax2_fused_2_5), 64i64))
                    tir.reads([T_matmul_NT_intermediate_reindex_shared_dyn_1[v0_5, v1_5, v2_5]])
                    tir.writes([T_relu_intermediate_2[v1_5, v2_5]])
                    T_relu_intermediate_2[v1_5, v2_5] = max(T_matmul_NT_intermediate_reindex_shared_dyn_1[v0_5, v1_5, v2_5], 0f16)
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