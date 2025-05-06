#include <iostream>
#include <cstdlib>
#include <mma.h>  // Needed for nvcuda::wmma
#include <cstdio>
#include <vector>
#include <random>
#include <cuda_runtime.h>


#include <cuda_fp16.h>
__device__ __forceinline__ __half hmax(__half a, __half b) {
    return __hgt(a, b) ? a : b;
}

__device__ __forceinline__ unsigned int my_pack_half2(const __half a, const __half b) {
    union {
        struct { __half x, y; } h2;
        unsigned int u32;
    } tmp;
    tmp.h2.x = a;
    tmp.h2.y = b;
    return tmp.u32;
}
// using namespace nvcuda;
// 已有的kernel定义
__device__ void Fused_0_fused_dense_relu_0(half* __restrict__ input0, half* __restrict__ param_0, half* __restrict__ T_relu_intermediate, char* shared) {
  int __flatten_tid = threadIdx.x;
  const dim3 threadIdx(__flatten_tid % 32, __flatten_tid / 32, 0);
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> T_matmul_NT_intermediate_wmma_accumulator[8];
  half* input0_shared = (half*)(shared+0);
  half* param_0_shared = (half*)(shared+18432);
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> input0_shared_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> param_0_shared_wmma_matrix_b[2];
  for (int i_c_outer_init = 0; i_c_outer_init < 4; ++i_c_outer_init) {
    for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
      nvcuda::wmma::fill_fragment(T_matmul_NT_intermediate_wmma_accumulator[((i_c_outer_init * 2) + j_c_outer_init)], __float2half_rn(0.000000e+00f));
    }
  }
  for (int k_outer = 0; k_outer < 2; ++k_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer < 4; ++ax0_ax1_fused_outer_outer_outer) {
      *(uint4*)(input0_shared + ((((ax0_ax1_fused_outer_outer_outer * 1280) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(input0 + ((((((((int)blockIdx.x) * 8192) + (ax0_ax1_fused_outer_outer_outer * 2048)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (k_outer * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    }
    for (int ax0_ax1_fused_outer_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_outer_1 < 2; ++ax0_ax1_fused_outer_outer_outer_1) {
      *(uint4*)(param_0_shared + ((((ax0_ax1_fused_outer_outer_outer_1 * 1280) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(param_0 + (((((ax0_ax1_fused_outer_outer_outer_1 * 2048) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (k_outer * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    }
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 2; ++k_inner_outer) {
      for (int ax0_outer = 0; ax0_outer < 4; ++ax0_outer) {
        nvcuda::wmma::load_matrix_sync(input0_shared_wmma_matrix_a[ax0_outer], (&(input0_shared[((((((int)threadIdx.y) >> 1) * 2560) + (ax0_outer * 640)) + (k_inner_outer * 16))])), 40);
      }
      for (int ax0_outer_1 = 0; ax0_outer_1 < 2; ++ax0_outer_1) {
        nvcuda::wmma::load_matrix_sync(param_0_shared_wmma_matrix_b[ax0_outer_1], (&(param_0_shared[((((((int)threadIdx.y) & 1) * 1280) + (ax0_outer_1 * 640)) + (k_inner_outer * 16))])), 40);
      }
      for (int i_c_outer = 0; i_c_outer < 4; ++i_c_outer) {
        for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
          nvcuda::wmma::mma_sync(T_matmul_NT_intermediate_wmma_accumulator[((i_c_outer * 2) + j_c_outer)], input0_shared_wmma_matrix_a[i_c_outer], param_0_shared_wmma_matrix_b[j_c_outer], T_matmul_NT_intermediate_wmma_accumulator[((i_c_outer * 2) + j_c_outer)]);
        }
      }
    }
  }
  __syncthreads();
  for (int ax0_inner_outer = 0; ax0_inner_outer < 4; ++ax0_inner_outer) {
    for (int ax1_inner_outer = 0; ax1_inner_outer < 2; ++ax1_inner_outer) {
      nvcuda::wmma::store_matrix_sync((&(input0_shared[(((((((int)threadIdx.y) >> 1) * 4608) + (ax0_inner_outer * 1152)) + ((((int)threadIdx.y) & 1) * 32)) + (ax1_inner_outer * 16))])), T_matmul_NT_intermediate_wmma_accumulator[((ax0_inner_outer * 2) + ax1_inner_outer)], 72, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer < 8; ++i_inner_j_inner_fused_outer_outer_outer) {
    uint4 __1;
      uint4 v_ = *(uint4*)(input0_shared + ((((i_inner_j_inner_fused_outer_outer_outer * 1152) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)));
      uint4 v__1 = make_uint4(my_pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), my_pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), my_pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), my_pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      ((half2*)(&(__1.x)))->x = hmax(((half2*)(&(v_.x)))->x, ((half2*)(&(v__1.x)))->x);
      ((half2*)(&(__1.x)))->y = hmax(((half2*)(&(v_.x)))->y, ((half2*)(&(v__1.x)))->y);
      ((half2*)(&(__1.y)))->x = hmax(((half2*)(&(v_.y)))->x, ((half2*)(&(v__1.y)))->x);
      ((half2*)(&(__1.y)))->y = hmax(((half2*)(&(v_.y)))->y, ((half2*)(&(v__1.y)))->y);
      ((half2*)(&(__1.z)))->x = hmax(((half2*)(&(v_.z)))->x, ((half2*)(&(v__1.z)))->x);
      ((half2*)(&(__1.z)))->y = hmax(((half2*)(&(v_.z)))->y, ((half2*)(&(v__1.z)))->y);
      ((half2*)(&(__1.w)))->x = hmax(((half2*)(&(v_.w)))->x, ((half2*)(&(v__1.w)))->x);
      ((half2*)(&(__1.w)))->y = hmax(((half2*)(&(v_.w)))->y, ((half2*)(&(v__1.w)))->y);
    *(uint4*)(T_relu_intermediate + ((((i_inner_j_inner_fused_outer_outer_outer * 1152) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) = __1;
  }
  __syncthreads();
}

__device__ void Fused_1_fused_dense_relu_1(half* __restrict__ input0, half* __restrict__ param_0, half* __restrict__ T_relu_intermediate, char* shared) {
  int __flatten_tid = threadIdx.x;
  const dim3 threadIdx(__flatten_tid % 32, __flatten_tid / 32, 0);
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> T_matmul_NT_intermediate_wmma_accumulator[8];
  half* input0_shared = (half*)input0;
  half* param_0_shared = (half*)(shared+0);
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> input0_shared_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> param_0_shared_wmma_matrix_b[2];
  for (int i_c_outer_init = 0; i_c_outer_init < 4; ++i_c_outer_init) {
    for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
      nvcuda::wmma::fill_fragment(T_matmul_NT_intermediate_wmma_accumulator[((i_c_outer_init * 2) + j_c_outer_init)], __float2half_rn(0.000000e+00f));
    }
  }
  for (int k_outer = 0; k_outer < 2; ++k_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer < 2; ++ax0_ax1_fused_outer_outer_outer) {
      *(uint4*)(param_0_shared + ((((ax0_ax1_fused_outer_outer_outer * 1280) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(param_0 + (((((ax0_ax1_fused_outer_outer_outer * 2048) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (k_outer * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    }
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 2; ++k_inner_outer) {
      for (int ax0_outer = 0; ax0_outer < 4; ++ax0_outer) {
        nvcuda::wmma::load_matrix_sync(input0_shared_wmma_matrix_a[ax0_outer], (&(input0_shared[(((((((int)threadIdx.y) >> 1) * 4608) + (ax0_outer * 1152)) + (k_outer * 32)) + (k_inner_outer * 16))])), 72);
      }
      for (int ax0_outer_1 = 0; ax0_outer_1 < 2; ++ax0_outer_1) {
        nvcuda::wmma::load_matrix_sync(param_0_shared_wmma_matrix_b[ax0_outer_1], (&(param_0_shared[((((((int)threadIdx.y) & 1) * 1280) + (ax0_outer_1 * 640)) + (k_inner_outer * 16))])), 40);
      }
      for (int i_c_outer = 0; i_c_outer < 4; ++i_c_outer) {
        for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
          nvcuda::wmma::mma_sync(T_matmul_NT_intermediate_wmma_accumulator[((i_c_outer * 2) + j_c_outer)], input0_shared_wmma_matrix_a[i_c_outer], param_0_shared_wmma_matrix_b[j_c_outer], T_matmul_NT_intermediate_wmma_accumulator[((i_c_outer * 2) + j_c_outer)]);
        }
      }
    }
  }
  __syncthreads();
  for (int ax0_inner_outer = 0; ax0_inner_outer < 4; ++ax0_inner_outer) {
    for (int ax1_inner_outer = 0; ax1_inner_outer < 2; ++ax1_inner_outer) {
      nvcuda::wmma::store_matrix_sync((&(input0_shared[(((((((int)threadIdx.y) >> 1) * 4608) + (ax0_inner_outer * 1152)) + ((((int)threadIdx.y) & 1) * 32)) + (ax1_inner_outer * 16))])), T_matmul_NT_intermediate_wmma_accumulator[((ax0_inner_outer * 2) + ax1_inner_outer)], 72, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer < 8; ++i_inner_j_inner_fused_outer_outer_outer) {
    uint4 __1;
      uint4 v_ = *(uint4*)(input0_shared + ((((i_inner_j_inner_fused_outer_outer_outer * 1152) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)));
      uint4 v__1 = make_uint4(my_pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), my_pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), my_pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), my_pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      ((half2*)(&(__1.x)))->x = hmax(((half2*)(&(v_.x)))->x, ((half2*)(&(v__1.x)))->x);
      ((half2*)(&(__1.x)))->y = hmax(((half2*)(&(v_.x)))->y, ((half2*)(&(v__1.x)))->y);
      ((half2*)(&(__1.y)))->x = hmax(((half2*)(&(v_.y)))->x, ((half2*)(&(v__1.y)))->x);
      ((half2*)(&(__1.y)))->y = hmax(((half2*)(&(v_.y)))->y, ((half2*)(&(v__1.y)))->y);
      ((half2*)(&(__1.z)))->x = hmax(((half2*)(&(v_.z)))->x, ((half2*)(&(v__1.z)))->x);
      ((half2*)(&(__1.z)))->y = hmax(((half2*)(&(v_.z)))->y, ((half2*)(&(v__1.z)))->y);
      ((half2*)(&(__1.w)))->x = hmax(((half2*)(&(v_.w)))->x, ((half2*)(&(v__1.w)))->x);
      ((half2*)(&(__1.w)))->y = hmax(((half2*)(&(v_.w)))->y, ((half2*)(&(v__1.w)))->y);
    *(uint4*)(T_relu_intermediate + ((((((int)blockIdx.x) * 8192) + (i_inner_j_inner_fused_outer_outer_outer * 1024)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8))) = __1;
  }
}

__global__ void __launch_bounds__(128) Fused(half* input0, half* input1, half* input2, half* output0) {
  __shared__ char shared[23552];
  Fused_0_fused_dense_relu_0(input0, input1, (half*)(shared+0), shared+0);
  Fused_1_fused_dense_relu_1((half*)(shared+0), input2, output0, shared+18432);
}

// 检查CUDA错误的辅助函数
void checkCudaError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERROR(call) checkCudaError(call, __FILE__, __LINE__)

int main() {
    // 输入张量的尺寸
    const int batch_size = 2073600;
    const int in_features = 64;
    const int out_features = 64;
    
    // 计算内存大小
    size_t input0_size = batch_size * in_features * sizeof(half);           // (2073600, 64)
    size_t input1_size = in_features * out_features * sizeof(half);          // (64, 64)
    size_t input2_size = out_features * out_features * sizeof(half);         // (64, 64)
    size_t output0_size = batch_size * out_features * sizeof(half);          // (2073600, 64)
    
    // 主机内存分配
    half *h_input0, *h_input1, *h_input2, *h_output0;
    h_input0 = (half*)malloc(input0_size);
    h_input1 = (half*)malloc(input1_size);
    h_input2 = (half*)malloc(input2_size);
    h_output0 = (half*)malloc(output0_size);
    
    // 初始化随机数生成器
    srand(time(NULL));
    
    // 随机初始化输入数据
    for (size_t i = 0; i < batch_size * in_features; i++) {
        // 生成-1.0到1.0之间的随机浮点数
        float random_val = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_input0[i] = __float2half_rn(random_val);
    }
    
    for (size_t i = 0; i < in_features * out_features; i++) {
        // 生成-1.0到1.0之间的随机浮点数
        float random_val = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_input1[i] = __float2half_rn(random_val);
    }
    
    for (size_t i = 0; i < out_features * out_features; i++) {
        // 生成-1.0到1.0之间的随机浮点数
        float random_val = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_input2[i] = __float2half_rn(random_val);
    }
    
    // 设备内存分配
    half *d_input0, *d_input1, *d_input2, *d_output0;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input0, input0_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input1, input1_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input2, input2_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output0, output0_size));
    
    // 将数据从主机内存复制到设备内存
    CHECK_CUDA_ERROR(cudaMemcpy(d_input0, h_input0, input0_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input1, h_input1, input1_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input2, h_input2, input2_size, cudaMemcpyHostToDevice));
    
    // 计算网格和块的大小
    // 从kernel实现分析，每个block处理64个数据点，batch_size = 2073600
    // 每个block处理128个线程，每个线程处理64个half数据
    // 计算需要的block数量
    int blocks_needed = (batch_size * in_features + 8191) / 8192;  // 向上取整
    
    // 配置kernel执行参数
    dim3 grid(blocks_needed);  // 网格大小
    dim3 block(128);           // 块大小
    
    printf("Launching kernel with grid size (%d) and block size (%d)\n", blocks_needed, 128);
    
    // 启动kernel
    Fused<<<grid, block>>>(d_input0, d_input1, d_input2, d_output0);
    
    // 检查kernel执行是否成功
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // 将结果从设备内存复制回主机内存
    CHECK_CUDA_ERROR(cudaMemcpy(h_output0, d_output0, output0_size, cudaMemcpyDeviceToHost));
    
    // 打印部分结果（前10个元素）
    printf("Output first 10 elements:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", __half2float(h_output0[i]));
    }
    printf("\n");
    
    // 释放设备内存
    CHECK_CUDA_ERROR(cudaFree(d_input0));
    CHECK_CUDA_ERROR(cudaFree(d_input1));
    CHECK_CUDA_ERROR(cudaFree(d_input2));
    CHECK_CUDA_ERROR(cudaFree(d_output0));
    
    // 释放主机内存
    free(h_input0);
    free(h_input1);
    free(h_input2);
    free(h_output0);
    
    printf("Kernel execution completed successfully!\n");
    
    return 0;
}