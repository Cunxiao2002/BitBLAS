#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530)

typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__
#define TVM_ALIGNED(x) __attribute__ ((aligned(x)))
#define TVM_HALF_OPERATOR(RTYPE, OP)                              \
  TVM_XINLINE RTYPE operator OP (half a, half b) {                \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (half a, T b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (T a, half b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }

#define TVM_HALF_ASSIGNOP(AOP, OP)                                \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const T& a) {                    \
    return *this = half(float(*this) OP float(a));                \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const volatile T& a) volatile {  \
    return *this = half(float(*this) OP float(a));                \
  }

class TVM_ALIGNED(2) half {
 public:
  uint16_t half_;

  static TVM_XINLINE half Binary(uint16_t value) {
    half res;
    res.half_ = value;
    return res;
  }

  TVM_XINLINE half() {}

  TVM_XINLINE half(const float& value) { constructor(value); }
  TVM_XINLINE explicit half(const double& value) { constructor(value); }
  TVM_XINLINE explicit half(const int8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const int32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const long long& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint64_t& value) { constructor(value); }

  TVM_XINLINE operator float() const {                          \
    return float(half2float(half_));                            \
  }                                                             \
  TVM_XINLINE operator float() const volatile {                 \
    return float(half2float(half_));                            \
  }


  TVM_HALF_ASSIGNOP(+=, +)
  TVM_HALF_ASSIGNOP(-=, -)
  TVM_HALF_ASSIGNOP(*=, *)
  TVM_HALF_ASSIGNOP(/=, /)

  TVM_XINLINE half operator+() {
    return *this;
  }

  TVM_XINLINE half operator-() {
    return half(-float(*this));
  }

  TVM_XINLINE half operator=(const half& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) {
    return *this = half(a);
  }

  TVM_XINLINE half operator=(const half& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) volatile {
    return *this = half(a);
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);   // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;   // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;   // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15;   // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;   // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;   // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;   // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;   // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  TVM_XINLINE uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  TVM_XINLINE uint16_t float2half(
    const volatile float& value) const volatile {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  TVM_XINLINE float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  TVM_XINLINE float half2float(
    const volatile uint16_t& value) const volatile {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  TVM_XINLINE void constructor(const T& value) {
    half_ = float2half(float(value));
  }
};

TVM_HALF_OPERATOR(half, +)
TVM_HALF_OPERATOR(half, -)
TVM_HALF_OPERATOR(half, *)
TVM_HALF_OPERATOR(half, /)
TVM_HALF_OPERATOR(bool, >)
TVM_HALF_OPERATOR(bool, <)
TVM_HALF_OPERATOR(bool, >=)
TVM_HALF_OPERATOR(bool, <=)

TVM_XINLINE half __float2half_rn(const float a) {
  return half(a);
}
#else
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#endif


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x, half y) {   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x) {          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

// Some fp16 math functions are not supported in cuda_fp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 530)
CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)
#else
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hexp, exp)
#endif
#endif

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY

struct __align__(8) half4 {
  __half x, y, z, w;
  __host__ __device__ half4() : x(__half(0)), y(__half(0)), z(__half(0)), w(__half(0)) {}
  __host__ __device__ half4(__half x, __half y, __half z, __half w) : x(x), y(y), z(z), w(w) {}

};
__host__ __device__ half4 make_half4(__half x, __half y, __half z, __half w) {
    return half4(x, y, z, w);
}
__forceinline__ __device__ unsigned int
cast_smem_ptr_to_int(const void* const smem_ptr)
{
  unsigned int smem_int;
  asm volatile ("{ .reg .u64 smem_int; cvta.to.shared.u64 smem_int, %1; cvt.u32.u64 %0, smem_int; }"
    : "=r"(smem_int) : "l"(smem_ptr));
  return smem_int;
}

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800) 
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1
#else
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 0
#endif
extern "C" __global__ void __launch_bounds__(128) fused_fused_dense_relu_fused_dense_relu_kernel(half* __restrict__ T_relu_intermediate_intermediate_1, half* __restrict__ input, half* __restrict__ param_func0, half* __restrict__ param_func1);
extern "C" __global__ void __launch_bounds__(128) fused_fused_dense_relu_fused_dense_relu_kernel(half* __restrict__ T_relu_intermediate_intermediate_1, half* __restrict__ input, half* __restrict__ param_func0, half* __restrict__ param_func1) {
  extern __shared__ uchar buf_dyn_shmem[];
  half T_matmul_NT_intermediate_reindex_shared_dyn_warp[128];
  half input0_reindex_shared_dyn_warp[32];
  half param_0_reindex_shared_dyn_warp[16];
  half T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[64];
  half param_0_reindex_shared_dyn_warp_1[16];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 4; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 2; ++ax2_0_3_init) {
      for (int i = 0; i < 8; ++i) {
T_matmul_NT_intermediate_reindex_shared_dyn_warp[(((((int64_t)ax1_0_3_init) * (int64_t)32) + (((int64_t)threadIdx.z) * (int64_t)16)) + (((int64_t)ax2_0_3_init) * (int64_t)8)) + i] = 0.0;}
;
    }
  }
  for (int ax3_0_0 = 0; ax3_0_0 < 2; ++ax3_0_0) {
    __syncthreads();
    #pragma unroll
    for (int64_t ax0_ax1_ax2_fused_2 = 0; ax0_ax1_ax2_fused_2 < (int64_t)4; ++ax0_ax1_ax2_fused_2) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((((int64_t)threadIdx.y) * (int64_t)2048) + (((int64_t)threadIdx.z) * (int64_t)1024)) + (ax0_ax1_ax2_fused_2 * (int64_t)256)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)32)) + (((((int64_t)threadIdx.x) & (int64_t)3) ^ (((int64_t)threadIdx.x) >> (int64_t)3)) * (int64_t)8)) + (int64_t)2048)) = *(uint4*)(input + (((((((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)threadIdx.y) * (int64_t)4096)) + (((int64_t)threadIdx.z) * (int64_t)2048)) + (ax0_ax1_ax2_fused_2 * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)64)) + (((int64_t)ax3_0_0) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)));
    }
    #pragma unroll
    for (int64_t ax0_ax1_ax2_fused_2_1 = 0; ax0_ax1_ax2_fused_2_1 < (int64_t)2; ++ax0_ax1_ax2_fused_2_1) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((((int64_t)threadIdx.y) * (int64_t)1024) + (((int64_t)threadIdx.z) * (int64_t)512)) + (ax0_ax1_ax2_fused_2_1 * (int64_t)256)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)32)) + (((((int64_t)threadIdx.x) & (int64_t)3) ^ (((int64_t)threadIdx.x) >> (int64_t)3)) * (int64_t)8)) + (int64_t)6144)) = *(uint4*)(param_func0 + ((((((((int64_t)threadIdx.y) * (int64_t)2048) + (((int64_t)threadIdx.z) * (int64_t)1024)) + (ax0_ax1_ax2_fused_2_1 * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)64)) + (((int64_t)ax3_0_0) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)));
    }
    __syncthreads();
    for (int64_t ax3_0_1 = 0; ax3_0_1 < (int64_t)2; ++ax3_0_1) {
      for (int64_t ax0_0 = 0; ax0_0 < (int64_t)4; ++ax0_0) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(((half*)buf_dyn_shmem)[(((((((int64_t)threadIdx.y) * (int64_t)2048) + (ax0_0 * (int64_t)512)) + ((((int64_t)threadIdx.x) & (int64_t)15) * (int64_t)32)) + ((((ax3_0_1 * (int64_t)2) + (((int64_t)threadIdx.x) >> (int64_t)4)) ^ ((((int64_t)threadIdx.x) & (int64_t)7) >> (int64_t)1)) * (int64_t)8)) + (int64_t)2048)])) + 0)));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[(((((((int64_t)threadIdx.y) * (int64_t)2048) + (ax0_0 * (int64_t)512)) + ((((int64_t)threadIdx.x) & (int64_t)15) * (int64_t)32)) + ((((ax3_0_1 * (int64_t)2) + (((int64_t)threadIdx.x) >> (int64_t)4)) ^ ((((int64_t)threadIdx.x) & (int64_t)7) >> (int64_t)1)) * (int64_t)8)) + (int64_t)2048)])) + 0))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(input0_reindex_shared_dyn_warp + (ax0_0 * (int64_t)8)))[0]), "=r"(((unsigned *)(input0_reindex_shared_dyn_warp + (ax0_0 * (int64_t)8)))[1]), "=r"(((unsigned *)(input0_reindex_shared_dyn_warp + (ax0_0 * (int64_t)8)))[2]), "=r"(((unsigned *)(input0_reindex_shared_dyn_warp + (ax0_0 * (int64_t)8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int64_t ax0_0_1 = 0; ax0_0_1 < (int64_t)2; ++ax0_0_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(((half*)buf_dyn_shmem)[((((((((int64_t)threadIdx.z) * (int64_t)1024) + (ax0_0_1 * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)4) * (int64_t)256)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)32)) + ((((ax3_0_1 * (int64_t)2) + ((((int64_t)threadIdx.x) & (int64_t)15) >> (int64_t)3)) ^ ((((int64_t)threadIdx.x) & (int64_t)7) >> (int64_t)1)) * (int64_t)8)) + (int64_t)6144)])) + 0)));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[((((((((int64_t)threadIdx.z) * (int64_t)1024) + (ax0_0_1 * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)4) * (int64_t)256)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)32)) + ((((ax3_0_1 * (int64_t)2) + ((((int64_t)threadIdx.x) & (int64_t)15) >> (int64_t)3)) ^ ((((int64_t)threadIdx.x) & (int64_t)7) >> (int64_t)1)) * (int64_t)8)) + (int64_t)6144)])) + 0))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(param_0_reindex_shared_dyn_warp + (ax0_0_1 * (int64_t)8)))[0]), "=r"(((unsigned *)(param_0_reindex_shared_dyn_warp + (ax0_0_1 * (int64_t)8)))[1]), "=r"(((unsigned *)(param_0_reindex_shared_dyn_warp + (ax0_0_1 * (int64_t)8)))[2]), "=r"(((unsigned *)(param_0_reindex_shared_dyn_warp + (ax0_0_1 * (int64_t)8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int ax1_0_3 = 0; ax1_0_3 < 4; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 2; ++ax2_0_3) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)threadIdx.z) * (int64_t)16)) + (((int64_t)ax2_0_3) * (int64_t)8))))[0]), "=r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)threadIdx.z) * (int64_t)16)) + (((int64_t)ax2_0_3) * (int64_t)8))))[1])
      : "r"(((unsigned *)((half*)input0_reindex_shared_dyn_warp + (((int64_t)ax1_0_3) * (int64_t)8)))[0]), "r"(((unsigned *)((half*)input0_reindex_shared_dyn_warp + (((int64_t)ax1_0_3) * (int64_t)8)))[1]), "r"(((unsigned *)((half*)input0_reindex_shared_dyn_warp + (((int64_t)ax1_0_3) * (int64_t)8)))[2]), "r"(((unsigned *)((half*)input0_reindex_shared_dyn_warp + (((int64_t)ax1_0_3) * (int64_t)8)))[3]), "r"(((unsigned *)((half*)param_0_reindex_shared_dyn_warp + (((int64_t)ax2_0_3) * (int64_t)8)))[0]), "r"(((unsigned *)((half*)param_0_reindex_shared_dyn_warp + (((int64_t)ax2_0_3) * (int64_t)8)))[1]), "r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)threadIdx.z) * (int64_t)16)) + (((int64_t)ax2_0_3) * (int64_t)8))))[0]), "r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)threadIdx.z) * (int64_t)16)) + (((int64_t)ax2_0_3) * (int64_t)8))))[1]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp + ((((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)threadIdx.z) * (int64_t)16)) + (((int64_t)ax2_0_3) * (int64_t)8)) + (int64_t)4)))[0]), "=r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp + ((((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)threadIdx.z) * (int64_t)16)) + (((int64_t)ax2_0_3) * (int64_t)8)) + (int64_t)4)))[1])
      : "r"(((unsigned *)((half*)input0_reindex_shared_dyn_warp + (((int64_t)ax1_0_3) * (int64_t)8)))[0]), "r"(((unsigned *)((half*)input0_reindex_shared_dyn_warp + (((int64_t)ax1_0_3) * (int64_t)8)))[1]), "r"(((unsigned *)((half*)input0_reindex_shared_dyn_warp + (((int64_t)ax1_0_3) * (int64_t)8)))[2]), "r"(((unsigned *)((half*)input0_reindex_shared_dyn_warp + (((int64_t)ax1_0_3) * (int64_t)8)))[3]), "r"(((unsigned *)((half*)param_0_reindex_shared_dyn_warp + ((((int64_t)ax2_0_3) * (int64_t)8) + (int64_t)4)))[0]), "r"(((unsigned *)((half*)param_0_reindex_shared_dyn_warp + ((((int64_t)ax2_0_3) * (int64_t)8) + (int64_t)4)))[1]), "r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp + ((((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)threadIdx.z) * (int64_t)16)) + (((int64_t)ax2_0_3) * (int64_t)8)) + (int64_t)4)))[0]), "r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp + ((((((int64_t)ax1_0_3) * (int64_t)32) + (((int64_t)threadIdx.z) * (int64_t)16)) + (((int64_t)ax2_0_3) * (int64_t)8)) + (int64_t)4)))[1]));
  }
        }
      }
    }
  }
  __syncthreads();
  for (int ax0_0_2 = 0; ax0_0_2 < 4; ++ax0_0_2) {
    for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
      for (int local_id = 0; local_id < 8; local_id+=2) {
*((uint *)&(&(((half*)buf_dyn_shmem)[(((((((int64_t)threadIdx.y) * (int64_t)4096) + (((int64_t)ax0_0_2) * (int64_t)1024)) + (((int64_t)threadIdx.z) * (int64_t)32)) + (((int64_t)ax1_0) * (int64_t)16)) + (int64_t)2048)]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 64) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))]) = *((uint *)&T_matmul_NT_intermediate_reindex_shared_dyn_warp[(((((int64_t)ax0_0_2) * (int64_t)32) + (((int64_t)threadIdx.z) * (int64_t)16)) + (((int64_t)ax1_0) * (int64_t)8)) + local_id]);
}
;
    }
  }
  #pragma unroll
  for (int ax0_ax1_ax2_fused_0 = 0; ax0_ax1_ax2_fused_0 < 16; ++ax0_ax1_ax2_fused_0) {
    __syncthreads();
    uint4 __1;
      uint4 v_ = *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)threadIdx.y) * (int64_t)4096) + (((int64_t)ax0_ax1_ax2_fused_0) * (int64_t)256)) + (((int64_t)threadIdx.x) * (int64_t)8)) + (int64_t)2048));
      uint4 v__1 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      ((half2*)(&(__1.x)))->x = max(((half2*)(&(v_.x)))->x, ((half2*)(&(v__1.x)))->x);
      ((half2*)(&(__1.x)))->y = max(((half2*)(&(v_.x)))->y, ((half2*)(&(v__1.x)))->y);
      ((half2*)(&(__1.y)))->x = max(((half2*)(&(v_.y)))->x, ((half2*)(&(v__1.y)))->x);
      ((half2*)(&(__1.y)))->y = max(((half2*)(&(v_.y)))->y, ((half2*)(&(v__1.y)))->y);
      ((half2*)(&(__1.z)))->x = max(((half2*)(&(v_.z)))->x, ((half2*)(&(v__1.z)))->x);
      ((half2*)(&(__1.z)))->y = max(((half2*)(&(v_.z)))->y, ((half2*)(&(v__1.z)))->y);
      ((half2*)(&(__1.w)))->x = max(((half2*)(&(v_.w)))->x, ((half2*)(&(v__1.w)))->x);
      ((half2*)(&(__1.w)))->y = max(((half2*)(&(v_.w)))->y, ((half2*)(&(v__1.w)))->y);
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)threadIdx.y) * (int64_t)4096) + (((int64_t)ax0_ax1_ax2_fused_0) * (int64_t)256)) + (((int64_t)threadIdx.x) * (int64_t)8)) + (int64_t)2048)) = __1;
  }
  for (int ax1_0_3_init_1 = 0; ax1_0_3_init_1 < 4; ++ax1_0_3_init_1) {
    for (int ax2_0_3_init_1 = 0; ax2_0_3_init_1 < 2; ++ax2_0_3_init_1) {
      for (int i = 0; i < 8; ++i) {
T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[((ax1_0_3_init_1 * 16) + (ax2_0_3_init_1 * 8)) + i] = 0.0;}
;
    }
  }
  for (int ax3_0_0_1 = 0; ax3_0_0_1 < 2; ++ax3_0_0_1) {
    __syncthreads();
    #pragma unroll
    for (int64_t ax0_ax1_ax2_fused_2_2 = 0; ax0_ax1_ax2_fused_2_2 < (int64_t)2; ++ax0_ax1_ax2_fused_2_2) {
      *(uint4*)(((half*)buf_dyn_shmem) + (((((((int64_t)threadIdx.y) * (int64_t)1024) + (((int64_t)threadIdx.z) * (int64_t)512)) + (ax0_ax1_ax2_fused_2_2 * (int64_t)256)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)32)) + (((((int64_t)threadIdx.x) & (int64_t)3) ^ (((int64_t)threadIdx.x) >> (int64_t)3)) * (int64_t)8))) = *(uint4*)(param_func1 + ((((((((int64_t)threadIdx.y) * (int64_t)2048) + (((int64_t)threadIdx.z) * (int64_t)1024)) + (ax0_ax1_ax2_fused_2_2 * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)2) * (int64_t)64)) + (((int64_t)ax3_0_0_1) * (int64_t)32)) + ((((int64_t)threadIdx.x) & (int64_t)3) * (int64_t)8)));
    }
    __syncthreads();
    for (int64_t ax3_0_1_1 = 0; ax3_0_1_1 < (int64_t)2; ++ax3_0_1_1) {
      for (int ax0_0_3 = 0; ax0_0_3 < 4; ++ax0_0_3) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(((half*)buf_dyn_shmem)[(((((((int64_t)threadIdx.y) * (int64_t)4096) + (((int64_t)ax0_0_3) * (int64_t)1024)) + (((int64_t)ax3_0_0_1) * (int64_t)32)) + (ax3_0_1_1 * (int64_t)16)) + (int64_t)2048)])) + (((((int64_t)threadIdx.x) & (int64_t)15) * (int64_t)64) + ((((int64_t)threadIdx.x) >> (int64_t)4) * (int64_t)8)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[(((((((int64_t)threadIdx.y) * (int64_t)4096) + (((int64_t)ax0_0_3) * (int64_t)1024)) + (((int64_t)ax3_0_0_1) * (int64_t)32)) + (ax3_0_1_1 * (int64_t)16)) + (int64_t)2048)])) + (((((int64_t)threadIdx.x) & (int64_t)15) * (int64_t)64) + ((((int64_t)threadIdx.x) >> (int64_t)4) * (int64_t)8))))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax0_0_3) * (int64_t)32) + (((int64_t)ax3_0_0_1) * (int64_t)16)) + (ax3_0_1_1 * (int64_t)8))))[0]), "=r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax0_0_3) * (int64_t)32) + (((int64_t)ax3_0_0_1) * (int64_t)16)) + (ax3_0_1_1 * (int64_t)8))))[1]), "=r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax0_0_3) * (int64_t)32) + (((int64_t)ax3_0_0_1) * (int64_t)16)) + (ax3_0_1_1 * (int64_t)8))))[2]), "=r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax0_0_3) * (int64_t)32) + (((int64_t)ax3_0_0_1) * (int64_t)16)) + (ax3_0_1_1 * (int64_t)8))))[3])
      : "r"(addr)
    );
  }
      }
      for (int64_t ax0_0_4 = 0; ax0_0_4 < (int64_t)2; ++ax0_0_4) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(((half*)buf_dyn_shmem)[(((((((int64_t)threadIdx.z) * (int64_t)1024) + (ax0_0_4 * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)4) * (int64_t)256)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)32)) + ((((ax3_0_1_1 * (int64_t)2) + ((((int64_t)threadIdx.x) & (int64_t)15) >> (int64_t)3)) ^ ((((int64_t)threadIdx.x) & (int64_t)7) >> (int64_t)1)) * (int64_t)8))])) + 0)));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[(((((((int64_t)threadIdx.z) * (int64_t)1024) + (ax0_0_4 * (int64_t)512)) + ((((int64_t)threadIdx.x) >> (int64_t)4) * (int64_t)256)) + ((((int64_t)threadIdx.x) & (int64_t)7) * (int64_t)32)) + ((((ax3_0_1_1 * (int64_t)2) + ((((int64_t)threadIdx.x) & (int64_t)15) >> (int64_t)3)) ^ ((((int64_t)threadIdx.x) & (int64_t)7) >> (int64_t)1)) * (int64_t)8))])) + 0))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(param_0_reindex_shared_dyn_warp_1 + (ax0_0_4 * (int64_t)8)))[0]), "=r"(((unsigned *)(param_0_reindex_shared_dyn_warp_1 + (ax0_0_4 * (int64_t)8)))[1]), "=r"(((unsigned *)(param_0_reindex_shared_dyn_warp_1 + (ax0_0_4 * (int64_t)8)))[2]), "=r"(((unsigned *)(param_0_reindex_shared_dyn_warp_1 + (ax0_0_4 * (int64_t)8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int ax1_0_3_1 = 0; ax1_0_3_1 < 4; ++ax1_0_3_1) {
        for (int ax2_0_3_1 = 0; ax2_0_3_1 < 2; ++ax2_0_3_1) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1 + ((((int64_t)ax1_0_3_1) * (int64_t)16) + (((int64_t)ax2_0_3_1) * (int64_t)8))))[0]), "=r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1 + ((((int64_t)ax1_0_3_1) * (int64_t)16) + (((int64_t)ax2_0_3_1) * (int64_t)8))))[1])
      : "r"(((unsigned *)((half*)T_matmul_NT_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax3_0_0_1) * (int64_t)16)) + (ax3_0_1_1 * (int64_t)8))))[0]), "r"(((unsigned *)((half*)T_matmul_NT_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax3_0_0_1) * (int64_t)16)) + (ax3_0_1_1 * (int64_t)8))))[1]), "r"(((unsigned *)((half*)T_matmul_NT_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax3_0_0_1) * (int64_t)16)) + (ax3_0_1_1 * (int64_t)8))))[2]), "r"(((unsigned *)((half*)T_matmul_NT_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax3_0_0_1) * (int64_t)16)) + (ax3_0_1_1 * (int64_t)8))))[3]), "r"(((unsigned *)((half*)param_0_reindex_shared_dyn_warp_1 + (((int64_t)ax2_0_3_1) * (int64_t)8)))[0]), "r"(((unsigned *)((half*)param_0_reindex_shared_dyn_warp_1 + (((int64_t)ax2_0_3_1) * (int64_t)8)))[1]), "r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1 + ((((int64_t)ax1_0_3_1) * (int64_t)16) + (((int64_t)ax2_0_3_1) * (int64_t)8))))[0]), "r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1 + ((((int64_t)ax1_0_3_1) * (int64_t)16) + (((int64_t)ax2_0_3_1) * (int64_t)8))))[1]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1 + (((((int64_t)ax1_0_3_1) * (int64_t)16) + (((int64_t)ax2_0_3_1) * (int64_t)8)) + (int64_t)4)))[0]), "=r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1 + (((((int64_t)ax1_0_3_1) * (int64_t)16) + (((int64_t)ax2_0_3_1) * (int64_t)8)) + (int64_t)4)))[1])
      : "r"(((unsigned *)((half*)T_matmul_NT_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax3_0_0_1) * (int64_t)16)) + (ax3_0_1_1 * (int64_t)8))))[0]), "r"(((unsigned *)((half*)T_matmul_NT_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax3_0_0_1) * (int64_t)16)) + (ax3_0_1_1 * (int64_t)8))))[1]), "r"(((unsigned *)((half*)T_matmul_NT_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax3_0_0_1) * (int64_t)16)) + (ax3_0_1_1 * (int64_t)8))))[2]), "r"(((unsigned *)((half*)T_matmul_NT_intermediate_reindex_shared_dyn_warp + (((((int64_t)ax1_0_3_1) * (int64_t)32) + (((int64_t)ax3_0_0_1) * (int64_t)16)) + (ax3_0_1_1 * (int64_t)8))))[3]), "r"(((unsigned *)((half*)param_0_reindex_shared_dyn_warp_1 + ((((int64_t)ax2_0_3_1) * (int64_t)8) + (int64_t)4)))[0]), "r"(((unsigned *)((half*)param_0_reindex_shared_dyn_warp_1 + ((((int64_t)ax2_0_3_1) * (int64_t)8) + (int64_t)4)))[1]), "r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1 + (((((int64_t)ax1_0_3_1) * (int64_t)16) + (((int64_t)ax2_0_3_1) * (int64_t)8)) + (int64_t)4)))[0]), "r"(((unsigned *)(T_matmul_NT_intermediate_reindex_shared_dyn_warp_1 + (((((int64_t)ax1_0_3_1) * (int64_t)16) + (((int64_t)ax2_0_3_1) * (int64_t)8)) + (int64_t)4)))[1]));
  }
        }
      }
    }
  }
  __syncthreads();
  for (int ax0_0_5 = 0; ax0_0_5 < 4; ++ax0_0_5) {
    for (int ax1_0_1 = 0; ax1_0_1 < 2; ++ax1_0_1) {
      for (int local_id = 0; local_id < 8; local_id+=2) {
*((uint *)&(&(((half*)buf_dyn_shmem)[(((((((int64_t)threadIdx.y) * (int64_t)4096) + (((int64_t)ax0_0_5) * (int64_t)1024)) + (((int64_t)threadIdx.z) * (int64_t)32)) + (((int64_t)ax1_0_1) * (int64_t)16)) + (int64_t)2048)]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 64) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))]) = *((uint *)&T_matmul_NT_intermediate_reindex_shared_dyn_warp_1[((ax0_0_5 * 16) + (ax1_0_1 * 8)) + local_id]);
}
;
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_ax2_fused_0_1 = 0; ax0_ax1_ax2_fused_0_1 < 16; ++ax0_ax1_ax2_fused_0_1) {
    uint4 __2;
      uint4 v__2 = *(uint4*)(((half*)buf_dyn_shmem) + ((((((int64_t)threadIdx.y) * (int64_t)4096) + (((int64_t)ax0_ax1_ax2_fused_0_1) * (int64_t)256)) + (((int64_t)threadIdx.x) * (int64_t)8)) + (int64_t)2048));
      uint4 v__3 = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      ((half2*)(&(__2.x)))->x = max(((half2*)(&(v__2.x)))->x, ((half2*)(&(v__3.x)))->x);
      ((half2*)(&(__2.x)))->y = max(((half2*)(&(v__2.x)))->y, ((half2*)(&(v__3.x)))->y);
      ((half2*)(&(__2.y)))->x = max(((half2*)(&(v__2.y)))->x, ((half2*)(&(v__3.y)))->x);
      ((half2*)(&(__2.y)))->y = max(((half2*)(&(v__2.y)))->y, ((half2*)(&(v__3.y)))->y);
      ((half2*)(&(__2.z)))->x = max(((half2*)(&(v__2.z)))->x, ((half2*)(&(v__3.z)))->x);
      ((half2*)(&(__2.z)))->y = max(((half2*)(&(v__2.z)))->y, ((half2*)(&(v__3.z)))->y);
      ((half2*)(&(__2.w)))->x = max(((half2*)(&(v__2.w)))->x, ((half2*)(&(v__3.w)))->x);
      ((half2*)(&(__2.w)))->y = max(((half2*)(&(v__2.w)))->y, ((half2*)(&(v__3.w)))->y);
    *(uint4*)(T_relu_intermediate_intermediate_1 + ((((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)threadIdx.y) * (int64_t)4096)) + (((int64_t)ax0_ax1_ax2_fused_0_1) * (int64_t)256)) + (((int64_t)threadIdx.x) * (int64_t)8))) = __2;
  }
}


extern "C" __global__ void fused_fused_dense_relu_fused_dense_relu_kernel(
    half* __restrict__ T_relu_intermediate_intermediate_1, 
    half* __restrict__ input, 
    half* __restrict__ param_func0, 
    half* __restrict__ param_func1);

// Utility function to check CUDA errors
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at: %s:%d\n", file, line);
        fprintf(stderr, "%s %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

int main() {
    // Define dimensions based on the provided information
    const int batch_size = 2073600;
    const int input_dim = 64;
    const int hidden_dim = 64;
    const int output_dim = 64;
    
    // Set up CUDA grid and block dimensions
    // Set up CUDA grid and block dimensions based on the given information
    dim3 gridDim(1, 16200, 1);
    dim3 blockDim(32, 2, 2);  // Block dimensions with 128 threads per block
    
    // Calculate memory sizes
    const size_t input_size = batch_size * input_dim;
    const size_t param0_size = input_dim * hidden_dim;
    const size_t param1_size = hidden_dim * output_dim;
    const size_t output_size = batch_size * output_dim;
    
    // Allocate host memory
    half *h_input, *h_param0, *h_param1, *h_output;
    h_input = new half[input_size];
    h_param0 = new half[param0_size];
    h_param1 = new half[param1_size];
    h_output = new half[output_size];
    
    // Seed the random number generator
    srand(time(NULL));
    
    // Initialize input data with random values
    for (size_t i = 0; i < input_size; i++) {
        float random_value = (float)rand() / RAND_MAX * 2.0f - 1.0f; // Random values between -1 and 1
        h_input[i] = __float2half(random_value);
    }
    
    // Initialize parameter matrices with random values
    for (size_t i = 0; i < param0_size; i++) {
        float random_value = (float)rand() / RAND_MAX * 0.2f - 0.1f; // Random values between -0.1 and 0.1
        h_param0[i] = __float2half(random_value);
    }
    
    for (size_t i = 0; i < param1_size; i++) {
        float random_value = (float)rand() / RAND_MAX * 0.2f - 0.1f; // Random values between -0.1 and 0.1
        h_param1[i] = __float2half(random_value);
    }
    
    // Allocate device memory
    half *d_input, *d_param0, *d_param1, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_param0, param0_size * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_param1, param1_size * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size * sizeof(half)));
    
    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, input_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_param0, h_param0, param0_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_param1, h_param1, param1_size * sizeof(half), cudaMemcpyHostToDevice));
    
    // Calculate shared memory size - kernel needs sufficient shared memory
    // Based on the error reports, we need to allocate more shared memory
    // Looking at the invalid access patterns (addresses around 0x4000-0x7FFF)
    size_t sharedMemSize = 20 * 1024; // 32KB of shared memory
    
    printf("Launching kernel with grid(%d,%d,%d), block(%d,%d,%d), shared memory size %zu bytes\n", 
           gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMemSize);
    
    // Check if the device has enough shared memory per block
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Device shared memory per block: %zu bytes\n", deviceProp.sharedMemPerBlock);
    printf("Required shared memory: %zu bytes\n", sharedMemSize);
    
    if (sharedMemSize > deviceProp.sharedMemPerBlock) {
        printf("Warning: Requested shared memory (%zu bytes) exceeds device limit (%zu bytes)\n", 
               sharedMemSize, deviceProp.sharedMemPerBlock);
        // Adjust to maximum available if needed
        sharedMemSize = deviceProp.sharedMemPerBlock;
        printf("Adjusted shared memory to maximum available: %zu bytes\n", sharedMemSize);
    }
    
    // Launch the kernel with adjusted shared memory size
    fused_fused_dense_relu_fused_dense_relu_kernel<<<gridDim, blockDim, sharedMemSize>>>(
        d_output, d_input, d_param0, d_param1);
    
    // Check for errors during kernel execution
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, output_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Print a few results for verification
    printf("Output samples (first few values):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", __half2float(h_output[i]));
    }
    printf("\n");
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_param0);
    cudaFree(d_param1);
    cudaFree(d_output);
    
    delete[] h_input;
    delete[] h_param0;
    delete[] h_param1;
    delete[] h_output;
    
    printf("Kernel execution completed successfully\n");
    
    return 0;
}