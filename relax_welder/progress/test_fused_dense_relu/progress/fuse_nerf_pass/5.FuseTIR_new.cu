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

@fused_fused_dense_relu_0_fused_dense_relu_1 = primfn(p_input0: handle, p_param_0: handle, p_param_1: handle, p_output0: handle) -> ()
  attr = {"tir.noalias": True}
  buffers = {input0_4: Buffer(input0_5: Pointer(float16), float16, [2073600i64, 64i64], []),
             param_0_4: Buffer(param_0_5: Pointer(float16), float16, [64i64, 64i64], []),
             param_1: Buffer(param_1_1: Pointer(float16), float16, [64i64, 64i64], []),
             T_relu_intermediate_intermediate_1: Buffer(T_relu_intermediate_4: Pointer(global float16), float16, [2073600i64, 64i64], [])}
  buffer_map = {p_input0: input0_4, p_param_0: param_0_4, p_param_1: param_1, p_output0: T_relu_intermediate_intermediate_1} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    input0_reindex_shared_dyn_2 = alloc_buffer(float16[1i64, 2073600i64, 64i64])
    param_0_reindex_shared_dyn_2 = alloc_buffer(float16[1i64, 64i64, 64i64])
    input0_reindex_shared_dyn_warp_2 = alloc_buffer(float16[1i64, 129600i64, 4i64, 32i64, 8i64])
    param_0_reindex_shared_dyn_warp_2 = alloc_buffer(float16[1i64, 4i64, 4i64, 32i64, 8i64])
    T_matmul_NT_intermediate_reindex_shared_dyn_2 = alloc_buffer(float16[1i64, 2073600i64, 64i64])
    T_matmul_NT_intermediate_reindex_shared_dyn_warp_2 = alloc_buffer(float16[1i64, 129600i64, 4i64, 32i64, 8i64])
    T_relu_intermediate_intermediate = alloc_buffer(float16[2073600i64, 64i64])
    input0_reindex_shared_dyn_3 = alloc_buffer(float16[1i64, 2073600i64, 64i64])
    param_0_reindex_shared_dyn_3 = alloc_buffer(float16[1i64, 64i64, 64i64])
    input0_reindex_shared_dyn_warp_3 = alloc_buffer(float16[1i64, 129600i64, 4i64, 32i64, 8i64])
    param_0_reindex_shared_dyn_warp_3 = alloc_buffer(float16[1i64, 4i64, 4i64, 32i64, 8i64])
    T_matmul_NT_intermediate_reindex_shared_dyn_3 = alloc_buffer(float16[1i64, 2073600i64, 64i64])
    T_matmul_NT_intermediate_reindex_shared_dyn_warp_3 = alloc_buffer(float16[1i64, 129600i64, 4i64, 32i64, 8i64])
     {
      for (ax0_2: int64, 0i64, 1i64) "thread_binding" {
        for (ax1_0_0_ax2_0_0_fused_2: int64, 0i64, 21600i64) "thread_binding" {
          for (ax1_0_1_ax2_0_1_fused_2: int64, 0i64, 1i64) "thread_binding" {
            for (ax1_0_2_2: int64, 0i64, 2i64) "thread_binding" {
              for (ax2_0_2_2: int64, 0i64, 2i64) "thread_binding" {
                for (ax1_0_3_init_2: int64, 0i64, 3i64) {
                  for (ax2_0_3_init_2: int64, 0i64, 2i64) {
                    block([1i64, 129600i64, 4i64], "T_matmul_NT_o_init") as [v0_o_10, v1_o_10, v2_o_10] {
                      bind(v0_o_10, ax0_2)
                      bind(v1_o_10, (((ax1_0_0_ax2_0_0_fused_2*6i64) + (ax1_0_2_2*3i64)) + ax1_0_3_init_2))
                      bind(v2_o_10, ((ax2_0_2_2*2i64) + ax2_0_3_init_2))
                      tir.reads([])
                      tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn_warp_2[0i64, v1_o_10, v2_o_10, 0i64:32i64, 0i64:8i64]])
                      block([1i64, 1i64], "T_matmul_NT_init_o") as [v1_i_init_o_2, v2_i_init_o_2] {
                        bind(v1_i_init_o_2, 0i64)
                        bind(v2_i_init_o_2, 0i64)
                        tir.reads([])
                        tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn_warp_2[0i64, v1_o_10, v2_o_10, 0i64:32i64, 0i64:8i64]])
                        C_warp_8 = match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_warp_2[0i64, v1_o_10, v2_o_10, 0i64:32i64, 0i64:8i64])
                        for (tx_10: int64, 0i64, 32i64) "thread_binding" {
                          @tir.mma_fill(8, C_warp_9: Pointer(warp float16), elem_offset_20: int64, dtype=float16)
                        }
                  }
                }
                for (ax3_0_0_2: int64, 0i64, 2i64) {
                  for (ax0_ax1_ax2_fused_0_6: int64, 0i64, 2i64) "thread_binding" {
                    for (ax0_ax1_ax2_fused_1_6: int64, 0i64, 2i64) "thread_binding" {
                      for (ax0_ax1_ax2_fused_2_6: int64, 0i64, 3i64) "unroll" {
                        for (ax0_ax1_ax2_fused_3_4: int64, 0i64, 32i64) "thread_binding" {
                          for (ax0_ax1_ax2_fused_4_4: int64, 0i64, 8i64) "vectorized" {
                            block([1i64, 2073600i64, 64i64], "input0_reindex_shared.dyn") as [v0_6, v1_6, v2_6] {
                              bind(v0_6, 0i64)
                              bind(v1_6, ((ax1_0_0_ax2_0_0_fused_2*96i64) + floordiv((((((ax0_ax1_ax2_fused_0_6*1536i64) + (ax0_ax1_ax2_fused_1_6*768i64)) + (ax0_ax1_ax2_fused_2_6*256i64)) + (ax0_ax1_ax2_fused_3_4*8i64)) + ax0_ax1_ax2_fused_4_4), 32i64)))
                              bind(v2_6, ((ax3_0_0_2*32i64) + floormod((((((ax0_ax1_ax2_fused_0_6*1536i64) + (ax0_ax1_ax2_fused_1_6*768i64)) + (ax0_ax1_ax2_fused_2_6*256i64)) + (ax0_ax1_ax2_fused_3_4*8i64)) + ax0_ax1_ax2_fused_4_4), 32i64)))
                              tir.reads([input0_4[v1_6, v2_6]])
                              tir.writes([input0_reindex_shared_dyn_2[v0_6, v1_6, v2_6]])
                              tir.attrs({"permuted_layout": 1})
                              input0_reindex_shared_dyn_2[v0_6, v1_6, v2_6] = input0_4[v1_6, v2_6]
                          }
                        }
                      }
                    }
                  }
                  for (ax0_ax1_ax2_fused_0_7: int64, 0i64, 2i64) "thread_binding" {
                    for (ax0_ax1_ax2_fused_1_7: int64, 0i64, 2i64) "thread_binding" {
                      for (ax0_ax1_ax2_fused_2_7: int64, 0i64, 2i64) "unroll" {
                        for (ax0_ax1_ax2_fused_3_5: int64, 0i64, 32i64) "thread_binding" {
                          for (ax0_ax1_ax2_fused_4_5: int64, 0i64, 8i64) "vectorized" {
                            block([1i64, 64i64, 64i64], "param_0_reindex_shared.dyn") as [v0_7, v1_7, v2_7] {
                              bind(v0_7, 0i64)
                              bind(v1_7, floordiv((((((ax0_ax1_ax2_fused_0_7*1024i64) + (ax0_ax1_ax2_fused_1_7*512i64)) + (ax0_ax1_ax2_fused_2_7*256i64)) + (ax0_ax1_ax2_fused_3_5*8i64)) + ax0_ax1_ax2_fused_4_5), 32i64))
                              bind(v2_7, ((ax3_0_0_2*32i64) + floormod((((((ax0_ax1_ax2_fused_0_7*1024i64) + (ax0_ax1_ax2_fused_1_7*512i64)) + (ax0_ax1_ax2_fused_2_7*256i64)) + (ax0_ax1_ax2_fused_3_5*8i64)) + ax0_ax1_ax2_fused_4_5), 32i64)))
                              tir.reads([param_0_4[v1_7, v2_7]])
                              tir.writes([param_0_reindex_shared_dyn_2[v0_7, v1_7, v2_7]])
                              tir.attrs({"permuted_layout": 1})
                              param_0_reindex_shared_dyn_2[v0_7, v1_7, v2_7] = param_0_4[v1_7, v2_7]
                          }
                        }
                      }
                    }
                  }
                  for (ax3_0_1_2: int64, 0i64, 2i64) {
                    for (ax0_0_6: int64, 0i64, 3i64) {
                      for (ax1_0_8: int64, 0i64, 1i64) {
                        block([1i64, 129600i64, 4i64], "input0_reindex_shared.dyn_warp_o") as [v0_o_11, v1_o_11, v2_o_11] {
                          bind(v0_o_11, 0i64)
                          bind(v1_o_11, (((ax1_0_0_ax2_0_0_fused_2*6i64) + (ax1_0_2_2*3i64)) + ax0_0_6))
                          bind(v2_o_11, (((ax3_0_0_2*2i64) + ax3_0_1_2) + ax1_0_8))
                          tir.reads([input0_reindex_shared_dyn_2[v0_o_11, (v1_o_11*16i64):((v1_o_11*16i64) + 16i64), (v2_o_11*16i64):((v2_o_11*16i64) + 16i64)]])
                          tir.writes([input0_reindex_shared_dyn_warp_2[v0_o_11, v1_o_11, v2_o_11, 0i64:32i64, 0i64:8i64]])
                          tir.attrs({"permuted_layout": 1})
                          warp_8 = match_buffer(input0_reindex_shared_dyn_warp_2[v0_o_11, v1_o_11, v2_o_11, 0i64:32i64, 0i64:8i64])
                          shared_8 = match_buffer(input0_reindex_shared_dyn_2[v0_o_11, (v1_o_11*16i64):((v1_o_11*16i64) + 16i64), (v2_o_11*16i64):((v2_o_11*16i64) + 16i64)])
                          for (tx_11: int64, 0i64, 32i64) "thread_binding" {
                            @tir.ptx_ldmatrix(False, 4, ".b16", warp_9: Pointer(warp float16), (elem_offset_21: int64 + (8i64*tx_11)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_9: Pointer(shared.dyn float16), elem_offset_22: int64, (shared_s0_4: int64*16i64), 1, dtype=handle), ((shared_s0_4*floormod(tx_11, 16i64)) + (8i64*floordiv(tx_11, 16i64))), dtype=float16)
                          }
                      }
                    }
                    for (ax0_0_7: int64, 0i64, 2i64) {
                      for (ax1_0_9: int64, 0i64, 1i64) {
                        block([1i64, 4i64, 4i64], "param_0_reindex_shared.dyn_warp_o") as [v0_o_12, v1_o_12, v2_o_12] {
                          bind(v0_o_12, 0i64)
                          bind(v1_o_12, ((ax2_0_2_2*2i64) + ax0_0_7))
                          bind(v2_o_12, (((ax3_0_0_2*2i64) + ax3_0_1_2) + ax1_0_9))
                          tir.reads([param_0_reindex_shared_dyn_2[v0_o_12, (v1_o_12*16i64):((v1_o_12*16i64) + 16i64), (v2_o_12*16i64):((v2_o_12*16i64) + 16i64)]])
                          tir.writes([param_0_reindex_shared_dyn_warp_2[v0_o_12, v1_o_12, v2_o_12, 0i64:32i64, 0i64:8i64]])
                          tir.attrs({"permuted_layout": 1})
                          warp_10 = match_buffer(param_0_reindex_shared_dyn_warp_2[v0_o_12, v1_o_12, v2_o_12, 0i64:32i64, 0i64:8i64])
                          shared_10 = match_buffer(param_0_reindex_shared_dyn_2[v0_o_12, (v1_o_12*16i64):((v1_o_12*16i64) + 16i64), (v2_o_12*16i64):((v2_o_12*16i64) + 16i64)])
                          for (tx_12: int64, 0i64, 32i64) "thread_binding" {
                            @tir.ptx_ldmatrix(False, 4, ".b16", warp_11: Pointer(warp float16), (elem_offset_23: int64 + (8i64*tx_12)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_11: Pointer(shared.dyn float16), elem_offset_24: int64, (shared_s0_5: int64*16i64), 1, dtype=handle), ((((shared_s0_5*8i64)*floordiv(tx_12, 16i64)) + (shared_s0_5*floormod(tx_12, 8i64))) + (8i64*floordiv(floormod(tx_12, 16i64), 8i64))), dtype=float16)
                          }
                      }
                    }
                    for (ax1_0_3_2: int64, 0i64, 3i64) {
                      for (ax2_0_3_2: int64, 0i64, 2i64) {
                        block([1i64, 129600i64, 4i64, tir.reduce_axis(0i64, 4i64)], "T_matmul_NT_o_update") as [v0_o_13, v1_o_13, v2_o_13, v3_o_2] {
                          bind(v0_o_13, ax0_2)
                          bind(v1_o_13, (((ax1_0_0_ax2_0_0_fused_2*6i64) + (ax1_0_2_2*3i64)) + ax1_0_3_2))
                          bind(v2_o_13, ((ax2_0_2_2*2i64) + ax2_0_3_2))
                          bind(v3_o_2, ((ax3_0_0_2*2i64) + ax3_0_1_2))
                          tir.reads([T_matmul_NT_intermediate_reindex_shared_dyn_warp_2[0i64, v1_o_13, v2_o_13, 0i64:32i64, 0i64:8i64], input0_reindex_shared_dyn_warp_2[0i64, v1_o_13, v3_o_2, 0i64:32i64, 0i64:8i64], param_0_reindex_shared_dyn_warp_2[0i64, v2_o_13, v3_o_2, 0i64:32i64, 0i64:8i64]])
                          tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn_warp_2[0i64, v1_o_13, v2_o_13, 0i64:32i64, 0i64:8i64]])
                          block([1i64, 1i64, tir.reduce_axis(0i64, 1i64)], "T_matmul_NT_o") as [v1_i_o_2, v2_i_o_2, v3_i_o_2] {
                            bind(v1_i_o_2, 0i64)
                            bind(v2_i_o_2, 0i64)
                            bind(v3_i_o_2, 0i64)
                            tir.reads([T_matmul_NT_intermediate_reindex_shared_dyn_warp_2[0i64, v1_o_13, v2_o_13, 0i64:32i64, 0i64:8i64], input0_reindex_shared_dyn_warp_2[0i64, v1_o_13, v3_o_2, 0i64:32i64, 0i64:8i64], param_0_reindex_shared_dyn_warp_2[0i64, v2_o_13, v3_o_2, 0i64:32i64, 0i64:8i64]])
                            tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn_warp_2[0i64, v1_o_13, v2_o_13, 0i64:32i64, 0i64:8i64]])
                            A_4 = match_buffer(input0_reindex_shared_dyn_warp_2[0i64, v1_o_13, v3_o_2, 0i64:32i64, 0i64:8i64])
                            B_4 = match_buffer(param_0_reindex_shared_dyn_warp_2[0i64, v2_o_13, v3_o_2, 0i64:32i64, 0i64:8i64])
                            C_8 = match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_warp_2[0i64, v1_o_13, v2_o_13, 0i64:32i64, 0i64:8i64])
                            for (tx_13: int64, 0i64, 32i64) "thread_binding" {
                              @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_5: Pointer(warp float16), (elem_offset_25: int64 + (tx_13*8i64)), B_5: Pointer(warp float16), (elem_offset_26: int64 + (tx_13*8i64)), C_9: Pointer(warp float16), (elem_offset_27: int64 + (tx_13*8i64)), False, dtype=float16)
                              @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_5, (elem_offset_25 + (tx_13*8i64)), B_5, ((elem_offset_26 + (tx_13*8i64)) + 4i64), C_9, ((elem_offset_27 + (tx_13*8i64)) + 4i64), False, dtype=float16)
                            }
                      }
                    }
                  }
                }
                for (ax0_0_8: int64, 0i64, 3i64) {
                  for (ax1_0_10: int64, 0i64, 2i64) {
                    block([1i64, 129600i64, 4i64], "T_matmul_NT_intermediate_reindex_shared.dyn_warp_o") as [v0_o_14, v1_o_14, v2_o_14] {
                      bind(v0_o_14, 0i64)
                      bind(v1_o_14, (((ax1_0_0_ax2_0_0_fused_2*6i64) + (ax1_0_2_2*3i64)) + ax0_0_8))
                      bind(v2_o_14, ((ax2_0_2_2*2i64) + ax1_0_10))
                      tir.reads([T_matmul_NT_intermediate_reindex_shared_dyn_warp_2[v0_o_14, v1_o_14, v2_o_14, 0i64:32i64, 0i64:8i64]])
                      tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn_2[v0_o_14, (v1_o_14*16i64):((v1_o_14*16i64) + 16i64), (v2_o_14*16i64):((v2_o_14*16i64) + 16i64)]])
                      C_warp_10 = match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_warp_2[v0_o_14, v1_o_14, v2_o_14, 0i64:32i64, 0i64:8i64])
                      C_10 = match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_2[v0_o_14, (v1_o_14*16i64):((v1_o_14*16i64) + 16i64), (v2_o_14*16i64):((v2_o_14*16i64) + 16i64)])
                      for (tx_14: int64, 0i64, 32i64) "thread_binding" {
                        @tir.mma_store(16, 16, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), C_11: Pointer(shared.dyn float16), elem_offset_28: int64, (C_s0_2: int64*16i64), 2, dtype=handle), C_warp_11: Pointer(warp float16), elem_offset_29: int64, C_s0_2, dtype=float16)
                      }
                  }
                }
              }
              for (ax0_ax1_ax2_fused_0_8: int64, 0i64, 12i64) "unroll" {
                for (ax0_ax1_ax2_fused_1_8: int64, 0i64, 32i64) "thread_binding" {
                  for (ax0_ax1_ax2_fused_2_8: int64, 0i64, 8i64) "vectorized" {
                    block([1i64, 2073600i64, 64i64], "T_matmul_NT_intermediate_reindex_shared.dyn") as [v0_8, v1_8, v2_8] {
                      bind(v0_8, 0i64)
                      bind(v1_8, (((ax1_0_0_ax2_0_0_fused_2*96i64) + (ax1_0_2_2*48i64)) + floordiv((((ax0_ax1_ax2_fused_0_8*256i64) + (ax0_ax1_ax2_fused_1_8*8i64)) + ax0_ax1_ax2_fused_2_8), 64i64)))
                      bind(v2_8, floormod((((ax0_ax1_ax2_fused_0_8*256i64) + (ax0_ax1_ax2_fused_1_8*8i64)) + ax0_ax1_ax2_fused_2_8), 64i64))
                      tir.reads([T_matmul_NT_intermediate_reindex_shared_dyn_2[v0_8, v1_8, v2_8]])
                      tir.writes([T_relu_intermediate_intermediate[v1_8, v2_8]])
                      T_relu_intermediate_intermediate[v1_8, v2_8] = max(T_matmul_NT_intermediate_reindex_shared_dyn_2[v0_8, v1_8, v2_8], 0f16)
                  }
                }
              }
            }
          }
        }
      }
      for (ax0_3: int64, 0i64, 1i64) "thread_binding" {
        for (ax1_0_0_ax2_0_0_fused_3: int64, 0i64, 21600i64) "thread_binding" {
          for (ax1_0_1_ax2_0_1_fused_3: int64, 0i64, 1i64) "thread_binding" {
            for (ax1_0_2_3: int64, 0i64, 2i64) "thread_binding" {
              for (ax2_0_2_3: int64, 0i64, 2i64) "thread_binding" {
                for (ax1_0_3_init_3: int64, 0i64, 3i64) {
                  for (ax2_0_3_init_3: int64, 0i64, 2i64) {
                    block([1i64, 129600i64, 4i64], "T_matmul_NT_o_init_1") as [v0_o_15, v1_o_15, v2_o_15] {
                      bind(v0_o_15, ax0_3)
                      bind(v1_o_15, (((ax1_0_0_ax2_0_0_fused_3*6i64) + (ax1_0_2_3*3i64)) + ax1_0_3_init_3))
                      bind(v2_o_15, ((ax2_0_2_3*2i64) + ax2_0_3_init_3))
                      tir.reads([])
                      tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn_warp_3[0i64, v1_o_15, v2_o_15, 0i64:32i64, 0i64:8i64]])
                      block([1i64, 1i64], "T_matmul_NT_init_o_1") as [v1_i_init_o_3, v2_i_init_o_3] {
                        bind(v1_i_init_o_3, 0i64)
                        bind(v2_i_init_o_3, 0i64)
                        tir.reads([])
                        tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn_warp_3[0i64, v1_o_15, v2_o_15, 0i64:32i64, 0i64:8i64]])
                        C_warp_12 = match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_warp_3[0i64, v1_o_15, v2_o_15, 0i64:32i64, 0i64:8i64])
                        for (tx_15: int64, 0i64, 32i64) "thread_binding" {
                          @tir.mma_fill(8, C_warp_13: Pointer(warp float16), elem_offset_30: int64, dtype=float16)
                        }
                  }
                }
                for (ax3_0_0_3: int64, 0i64, 2i64) {
                  for (ax0_ax1_ax2_fused_0_9: int64, 0i64, 2i64) "thread_binding" {
                    for (ax0_ax1_ax2_fused_1_9: int64, 0i64, 2i64) "thread_binding" {
                      for (ax0_ax1_ax2_fused_2_9: int64, 0i64, 3i64) "unroll" {
                        for (ax0_ax1_ax2_fused_3_6: int64, 0i64, 32i64) "thread_binding" {
                          for (ax0_ax1_ax2_fused_4_6: int64, 0i64, 8i64) "vectorized" {
                            block([1i64, 2073600i64, 64i64], "input0_reindex_shared.dyn_1") as [v0_9, v1_9, v2_9] {
                              bind(v0_9, 0i64)
                              bind(v1_9, ((ax1_0_0_ax2_0_0_fused_3*96i64) + floordiv((((((ax0_ax1_ax2_fused_0_9*1536i64) + (ax0_ax1_ax2_fused_1_9*768i64)) + (ax0_ax1_ax2_fused_2_9*256i64)) + (ax0_ax1_ax2_fused_3_6*8i64)) + ax0_ax1_ax2_fused_4_6), 32i64)))
                              bind(v2_9, ((ax3_0_0_3*32i64) + floormod((((((ax0_ax1_ax2_fused_0_9*1536i64) + (ax0_ax1_ax2_fused_1_9*768i64)) + (ax0_ax1_ax2_fused_2_9*256i64)) + (ax0_ax1_ax2_fused_3_6*8i64)) + ax0_ax1_ax2_fused_4_6), 32i64)))
                              tir.reads([T_relu_intermediate_intermediate[v1_9, v2_9]])
                              tir.writes([input0_reindex_shared_dyn_3[v0_9, v1_9, v2_9]])
                              tir.attrs({"permuted_layout": 1})
                              input0_reindex_shared_dyn_3[v0_9, v1_9, v2_9] = T_relu_intermediate_intermediate[v1_9, v2_9]
                          }
                        }
                      }
                    }
                  }
                  for (ax0_ax1_ax2_fused_0_10: int64, 0i64, 2i64) "thread_binding" {
                    for (ax0_ax1_ax2_fused_1_10: int64, 0i64, 2i64) "thread_binding" {
                      for (ax0_ax1_ax2_fused_2_10: int64, 0i64, 2i64) "unroll" {
                        for (ax0_ax1_ax2_fused_3_7: int64, 0i64, 32i64) "thread_binding" {
                          for (ax0_ax1_ax2_fused_4_7: int64, 0i64, 8i64) "vectorized" {
                            block([1i64, 64i64, 64i64], "param_0_reindex_shared.dyn_1") as [v0_10, v1_10, v2_10] {
                              bind(v0_10, 0i64)
                              bind(v1_10, floordiv((((((ax0_ax1_ax2_fused_0_10*1024i64) + (ax0_ax1_ax2_fused_1_10*512i64)) + (ax0_ax1_ax2_fused_2_10*256i64)) + (ax0_ax1_ax2_fused_3_7*8i64)) + ax0_ax1_ax2_fused_4_7), 32i64))
                              bind(v2_10, ((ax3_0_0_3*32i64) + floormod((((((ax0_ax1_ax2_fused_0_10*1024i64) + (ax0_ax1_ax2_fused_1_10*512i64)) + (ax0_ax1_ax2_fused_2_10*256i64)) + (ax0_ax1_ax2_fused_3_7*8i64)) + ax0_ax1_ax2_fused_4_7), 32i64)))
                              tir.reads([param_1[v1_10, v2_10]])
                              tir.writes([param_0_reindex_shared_dyn_3[v0_10, v1_10, v2_10]])
                              tir.attrs({"permuted_layout": 1})
                              param_0_reindex_shared_dyn_3[v0_10, v1_10, v2_10] = param_1[v1_10, v2_10]
                          }
                        }
                      }
                    }
                  }
                  for (ax3_0_1_3: int64, 0i64, 2i64) {
                    for (ax0_0_9: int64, 0i64, 3i64) {
                      for (ax1_0_11: int64, 0i64, 1i64) {
                        block([1i64, 129600i64, 4i64], "input0_reindex_shared.dyn_warp_o_1") as [v0_o_16, v1_o_16, v2_o_16] {
                          bind(v0_o_16, 0i64)
                          bind(v1_o_16, (((ax1_0_0_ax2_0_0_fused_3*6i64) + (ax1_0_2_3*3i64)) + ax0_0_9))
                          bind(v2_o_16, (((ax3_0_0_3*2i64) + ax3_0_1_3) + ax1_0_11))
                          tir.reads([input0_reindex_shared_dyn_3[v0_o_16, (v1_o_16*16i64):((v1_o_16*16i64) + 16i64), (v2_o_16*16i64):((v2_o_16*16i64) + 16i64)]])
                          tir.writes([input0_reindex_shared_dyn_warp_3[v0_o_16, v1_o_16, v2_o_16, 0i64:32i64, 0i64:8i64]])
                          tir.attrs({"permuted_layout": 1})
                          warp_12 = match_buffer(input0_reindex_shared_dyn_warp_3[v0_o_16, v1_o_16, v2_o_16, 0i64:32i64, 0i64:8i64])
                          shared_12 = match_buffer(input0_reindex_shared_dyn_3[v0_o_16, (v1_o_16*16i64):((v1_o_16*16i64) + 16i64), (v2_o_16*16i64):((v2_o_16*16i64) + 16i64)])
                          for (tx_16: int64, 0i64, 32i64) "thread_binding" {
                            @tir.ptx_ldmatrix(False, 4, ".b16", warp_13: Pointer(warp float16), (elem_offset_31: int64 + (8i64*tx_16)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_13: Pointer(shared.dyn float16), elem_offset_32: int64, (shared_s0_6: int64*16i64), 1, dtype=handle), ((shared_s0_6*floormod(tx_16, 16i64)) + (8i64*floordiv(tx_16, 16i64))), dtype=float16)
                          }
                      }
                    }
                    for (ax0_0_10: int64, 0i64, 2i64) {
                      for (ax1_0_12: int64, 0i64, 1i64) {
                        block([1i64, 4i64, 4i64], "param_0_reindex_shared.dyn_warp_o_1") as [v0_o_17, v1_o_17, v2_o_17] {
                          bind(v0_o_17, 0i64)
                          bind(v1_o_17, ((ax2_0_2_3*2i64) + ax0_0_10))
                          bind(v2_o_17, (((ax3_0_0_3*2i64) + ax3_0_1_3) + ax1_0_12))
                          tir.reads([param_0_reindex_shared_dyn_3[v0_o_17, (v1_o_17*16i64):((v1_o_17*16i64) + 16i64), (v2_o_17*16i64):((v2_o_17*16i64) + 16i64)]])
                          tir.writes([param_0_reindex_shared_dyn_warp_3[v0_o_17, v1_o_17, v2_o_17, 0i64:32i64, 0i64:8i64]])
                          tir.attrs({"permuted_layout": 1})
                          warp_14 = match_buffer(param_0_reindex_shared_dyn_warp_3[v0_o_17, v1_o_17, v2_o_17, 0i64:32i64, 0i64:8i64])
                          shared_14 = match_buffer(param_0_reindex_shared_dyn_3[v0_o_17, (v1_o_17*16i64):((v1_o_17*16i64) + 16i64), (v2_o_17*16i64):((v2_o_17*16i64) + 16i64)])
                          for (tx_17: int64, 0i64, 32i64) "thread_binding" {
                            @tir.ptx_ldmatrix(False, 4, ".b16", warp_15: Pointer(warp float16), (elem_offset_33: int64 + (8i64*tx_17)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_15: Pointer(shared.dyn float16), elem_offset_34: int64, (shared_s0_7: int64*16i64), 1, dtype=handle), ((((shared_s0_7*8i64)*floordiv(tx_17, 16i64)) + (shared_s0_7*floormod(tx_17, 8i64))) + (8i64*floordiv(floormod(tx_17, 16i64), 8i64))), dtype=float16)
                          }
                      }
                    }
                    for (ax1_0_3_3: int64, 0i64, 3i64) {
                      for (ax2_0_3_3: int64, 0i64, 2i64) {
                        block([1i64, 129600i64, 4i64, tir.reduce_axis(0i64, 4i64)], "T_matmul_NT_o_update_1") as [v0_o_18, v1_o_18, v2_o_18, v3_o_3] {
                          bind(v0_o_18, ax0_3)
                          bind(v1_o_18, (((ax1_0_0_ax2_0_0_fused_3*6i64) + (ax1_0_2_3*3i64)) + ax1_0_3_3))
                          bind(v2_o_18, ((ax2_0_2_3*2i64) + ax2_0_3_3))
                          bind(v3_o_3, ((ax3_0_0_3*2i64) + ax3_0_1_3))
                          tir.reads([T_matmul_NT_intermediate_reindex_shared_dyn_warp_3[0i64, v1_o_18, v2_o_18, 0i64:32i64, 0i64:8i64], input0_reindex_shared_dyn_warp_3[0i64, v1_o_18, v3_o_3, 0i64:32i64, 0i64:8i64], param_0_reindex_shared_dyn_warp_3[0i64, v2_o_18, v3_o_3, 0i64:32i64, 0i64:8i64]])
                          tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn_warp_3[0i64, v1_o_18, v2_o_18, 0i64:32i64, 0i64:8i64]])
                          block([1i64, 1i64, tir.reduce_axis(0i64, 1i64)], "T_matmul_NT_o_1") as [v1_i_o_3, v2_i_o_3, v3_i_o_3] {
                            bind(v1_i_o_3, 0i64)
                            bind(v2_i_o_3, 0i64)
                            bind(v3_i_o_3, 0i64)
                            tir.reads([T_matmul_NT_intermediate_reindex_shared_dyn_warp_3[0i64, v1_o_18, v2_o_18, 0i64:32i64, 0i64:8i64], input0_reindex_shared_dyn_warp_3[0i64, v1_o_18, v3_o_3, 0i64:32i64, 0i64:8i64], param_0_reindex_shared_dyn_warp_3[0i64, v2_o_18, v3_o_3, 0i64:32i64, 0i64:8i64]])
                            tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn_warp_3[0i64, v1_o_18, v2_o_18, 0i64:32i64, 0i64:8i64]])
                            A_6 = match_buffer(input0_reindex_shared_dyn_warp_3[0i64, v1_o_18, v3_o_3, 0i64:32i64, 0i64:8i64])
                            B_6 = match_buffer(param_0_reindex_shared_dyn_warp_3[0i64, v2_o_18, v3_o_3, 0i64:32i64, 0i64:8i64])
                            C_12 = match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_warp_3[0i64, v1_o_18, v2_o_18, 0i64:32i64, 0i64:8i64])
                            for (tx_18: int64, 0i64, 32i64) "thread_binding" {
                              @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_7: Pointer(warp float16), (elem_offset_35: int64 + (tx_18*8i64)), B_7: Pointer(warp float16), (elem_offset_36: int64 + (tx_18*8i64)), C_13: Pointer(warp float16), (elem_offset_37: int64 + (tx_18*8i64)), False, dtype=float16)
                              @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_7, (elem_offset_35 + (tx_18*8i64)), B_7, ((elem_offset_36 + (tx_18*8i64)) + 4i64), C_13, ((elem_offset_37 + (tx_18*8i64)) + 4i64), False, dtype=float16)
                            }
                      }
                    }
                  }
                }
                for (ax0_0_11: int64, 0i64, 3i64) {
                  for (ax1_0_13: int64, 0i64, 2i64) {
                    block([1i64, 129600i64, 4i64], "T_matmul_NT_intermediate_reindex_shared.dyn_warp_o_1") as [v0_o_19, v1_o_19, v2_o_19] {
                      bind(v0_o_19, 0i64)
                      bind(v1_o_19, (((ax1_0_0_ax2_0_0_fused_3*6i64) + (ax1_0_2_3*3i64)) + ax0_0_11))
                      bind(v2_o_19, ((ax2_0_2_3*2i64) + ax1_0_13))
                      tir.reads([T_matmul_NT_intermediate_reindex_shared_dyn_warp_3[v0_o_19, v1_o_19, v2_o_19, 0i64:32i64, 0i64:8i64]])
                      tir.writes([T_matmul_NT_intermediate_reindex_shared_dyn_3[v0_o_19, (v1_o_19*16i64):((v1_o_19*16i64) + 16i64), (v2_o_19*16i64):((v2_o_19*16i64) + 16i64)]])
                      C_warp_14 = match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_warp_3[v0_o_19, v1_o_19, v2_o_19, 0i64:32i64, 0i64:8i64])
                      C_14 = match_buffer(T_matmul_NT_intermediate_reindex_shared_dyn_3[v0_o_19, (v1_o_19*16i64):((v1_o_19*16i64) + 16i64), (v2_o_19*16i64):((v2_o_19*16i64) + 16i64)])
                      for (tx_19: int64, 0i64, 32i64) "thread_binding" {
                        @tir.mma_store(16, 16, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), C_15: Pointer(shared.dyn float16), elem_offset_38: int64, (C_s0_3: int64*16i64), 2, dtype=handle), C_warp_15: Pointer(warp float16), elem_offset_39: int64, C_s0_3, dtype=float16)
                      }
                  }
                }
              }
              for (ax0_ax1_ax2_fused_0_11: int64, 0i64, 12i64) "unroll" {
                for (ax0_ax1_ax2_fused_1_11: int64, 0i64, 32i64) "thread_binding" {
                  for (ax0_ax1_ax2_fused_2_11: int64, 0i64, 8i64) "vectorized" {
                    block([1i64, 2073600i64, 64i64], "T_matmul_NT_intermediate_reindex_shared.dyn_1") as [v0_11, v1_11, v2_11] {
                      bind(v0_11, 0i64)
                      bind(v1_11, (((ax1_0_0_ax2_0_0_fused_3*96i64) + (ax1_0_2_3*48i64)) + floordiv((((ax0_ax1_ax2_fused_0_11*256i64) + (ax0_ax1_ax2_fused_1_11*8i64)) + ax0_ax1_ax2_fused_2_11), 64i64)))
                      bind(v2_11, floormod((((ax0_ax1_ax2_fused_0_11*256i64) + (ax0_ax1_ax2_fused_1_11*8i64)) + ax0_ax1_ax2_fused_2_11), 64i64))
                      tir.reads([T_matmul_NT_intermediate_reindex_shared_dyn_3[v0_11, v1_11, v2_11]])
                      tir.writes([T_relu_intermediate_intermediate_1[v1_11, v2_11]])
                      T_relu_intermediate_intermediate_1[v1_11, v2_11] = max(T_matmul_NT_intermediate_reindex_shared_dyn_3[v0_11, v1_11, v2_11], 0f16)
                  }
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