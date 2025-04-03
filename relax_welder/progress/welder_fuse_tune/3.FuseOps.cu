#[version = "0.0.5"]




@matmul = primfn(var_x: handle, var_w: handle, var_matmul: handle) -> ()
  attr = {"tir.noalias": True, "op_pattern": 0}
  buffers = {x: Buffer(x_1: Pointer(global float32), float32, [128i64, 128i64], []),
             w: Buffer(w_1: Pointer(global float32), float32, [128i64, 128i64], []),
             matmul: Buffer(matmul_1: Pointer(global float32), float32, [128i64, 128i64], [])}
  buffer_map = {var_x: x, var_w: w, var_matmul: matmul} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    for (i0: int64, 0i64, 128i64) {
      for (i1: int64, 0i64, 128i64) {
        for (k: int64, 0i64, 128i64) {
          block([128i64, 128i64, tir.reduce_axis(0i64, 128i64)], "matmul") as [v_i0, v_i1, v_k] {
            bind(v_i0, i0)
            bind(v_i1, i1)
            bind(v_k, k)
            tir.reads([x[v_i0, v_k], w[v_k, v_i1]])
            tir.writes([matmul[v_i0, v_i1]])
            with init() {
              matmul[v_i0, v_i1] = 0f32
            }
            matmul[v_i0, v_i1] = (matmul[v_i0, v_i1] + (x[v_i0, v_k]*w[v_k, v_i1]))
        }
      }
    }
}

@relu = primfn(var_lv0: handle, var_compute: handle) -> ()
  attr = {"tir.noalias": True, "op_pattern": 0}
  buffers = {lv0: Buffer(lv0_1: Pointer(global float32), float32, [128i64, 128i64], []),
             compute: Buffer(compute_1: Pointer(global float32), float32, [128i64, 128i64], [])}
  buffer_map = {var_lv0: lv0, var_compute: compute} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    for (i0_1: int64, 0i64, 128i64) {
      for (i1_1: int64, 0i64, 128i64) {
        block([128i64, 128i64], "compute") as [v_i0_1, v_i1_1] {
          bind(v_i0_1, i0_1)
          bind(v_i1_1, i1_1)
          tir.reads([lv0[v_i0_1, v_i1_1]])
          tir.writes([compute[v_i0_1, v_i1_1]])
          compute[v_i0_1, v_i1_1] = max(lv0[v_i0_1, v_i1_1], 0f32)
      }
    }
}

/* For debugging purposes the metadata section has been omitted.
 * If you would like to see the full metadata section you can set the 
 * option to `True` when invoking `astext`. 
 */