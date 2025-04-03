#[version = "0.0.5"]
@fused_matmul_relu = primfn(p_x: handle, p_w: handle, p_output0: handle) -> ()
  attr = {"tir.noalias": True}
  buffers = {x: Buffer(x_1: Pointer(float32), float32, [128i64, 128i64], []),
             w: Buffer(w_1: Pointer(float32), float32, [128i64, 128i64], []),
             compute_intermediate: Buffer(compute: Pointer(global float32), float32, [128i64, 128i64], [])}
  buffer_map = {p_x: x, p_w: w, p_output0: compute_intermediate} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    matmul_intermediate = alloc_buffer(float32[128i64, 128i64])
     {
      for (i0: int64, 0i64, 128i64) {
        for (i1: int64, 0i64, 128i64) {
          for (k: int64, 0i64, 128i64) {
            block([128i64, 128i64, tir.reduce_axis(0i64, 128i64)], "matmul") as [v_i0, v_i1, v_k] {
              bind(v_i0, i0)
              bind(v_i1, i1)
              bind(v_k, k)
              tir.reads([x[v_i0, v_k], w[v_k, v_i1]])
              tir.writes([matmul_intermediate[v_i0, v_i1]])
              with init() {
                matmul_intermediate[v_i0, v_i1] = 0f32
              }
              matmul_intermediate[v_i0, v_i1] = (matmul_intermediate[v_i0, v_i1] + (x[v_i0, v_k]*w[v_k, v_i1]))
          }
        }
      }
      for (i0_1: int64, 0i64, 128i64) {
        for (i1_1: int64, 0i64, 128i64) {
          block([128i64, 128i64], "compute") as [v_i0_1, v_i1_1] {
            bind(v_i0_1, i0_1)
            bind(v_i1_1, i1_1)
            tir.reads([matmul_intermediate[v_i0_1, v_i1_1]])
            tir.writes([compute_intermediate[v_i0_1, v_i1_1]])
            compute_intermediate[v_i0_1, v_i1_1] = max(matmul_intermediate[v_i0_1, v_i1_1], 0f32)
        }
      }
    }
}



/* For debugging purposes the metadata section has been omitted.
 * If you would like to see the full metadata section you can set the 
 * option to `True` when invoking `astext`. 
 */