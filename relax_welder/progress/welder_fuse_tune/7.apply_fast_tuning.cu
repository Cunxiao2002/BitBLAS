#[version = "0.0.5"]
@fused_matmul_relu = primfn(p_x: handle, p_w: handle, p_output0: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "fused_matmul_relu", "tir.is_scheduled": 1}
  buffers = {x: Buffer(x_1: Pointer(float32), float32, [128i64, 128i64], []),
             w: Buffer(w_1: Pointer(float32), float32, [128i64, 128i64], []),
             compute_intermediate: Buffer(compute: Pointer(global float32), float32, [128i64, 128i64], [])}
  buffer_map = {p_x: x, p_w: w, p_output0: compute_intermediate} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    matmul_intermediate_local = alloc_buffer(float32[128i64, 128i64])
    for (ax0_0_ax1_0_fused: int64, 0i64, 128i64) "thread_binding" {
      for (ax0_1_ax1_1_fused: int64, 0i64, 128i64) "thread_binding" {
        block([128i64, 128i64], "matmul_init") as [v0, v1] {
          bind(v0, ((floordiv(ax0_0_ax1_0_fused, 2i64)*2i64) + floordiv(ax0_1_ax1_1_fused, 64i64)))
          bind(v1, ((floormod(ax0_0_ax1_0_fused, 2i64)*64i64) + floormod(ax0_1_ax1_1_fused, 64i64)))
          tir.reads([])
          tir.writes([matmul_intermediate_local[v0, v1]])
          matmul_intermediate_local[v0, v1] = 0f32
        for (ax2_0: int64, 0i64, 2i64) {
          for (ax2_1: int64, 0i64, 64i64) {
            block([128i64, 128i64, tir.reduce_axis(0i64, 128i64)], "matmul_update") as [v0_1, v1_1, v2] {
              bind(v0_1, ((floordiv(ax0_0_ax1_0_fused, 2i64)*2i64) + floordiv(ax0_1_ax1_1_fused, 64i64)))
              bind(v1_1, ((floormod(ax0_0_ax1_0_fused, 2i64)*64i64) + floormod(ax0_1_ax1_1_fused, 64i64)))
              bind(v2, ((ax2_0*64i64) + ax2_1))
              tir.reads([matmul_intermediate_local[v0_1, v1_1], x[v0_1, v2], w[v2, v1_1]])
              tir.writes([matmul_intermediate_local[v0_1, v1_1]])
              matmul_intermediate_local[v0_1, v1_1] = (matmul_intermediate_local[v0_1, v1_1] + (x[v0_1, v2]*w[v2, v1_1]))
          }
        }
        block([128i64, 128i64], "matmul_intermediate_local") as [v0_2, v1_2] {
          bind(v0_2, ((floordiv(ax0_0_ax1_0_fused, 2i64)*2i64) + floordiv(ax0_1_ax1_1_fused, 64i64)))
          bind(v1_2, ((floormod(ax0_0_ax1_0_fused, 2i64)*64i64) + floormod(ax0_1_ax1_1_fused, 64i64)))
          tir.reads([matmul_intermediate_local[v0_2, v1_2]])
          tir.writes([compute_intermediate[v0_2, v1_2]])
          compute_intermediate[v0_2, v1_2] = max(matmul_intermediate_local[v0_2, v1_2], 0f32)
      }
    }
}



/* For debugging purposes the metadata section has been omitted.
 * If you would like to see the full metadata section you can set the 
 * option to `True` when invoking `astext`. 
 */