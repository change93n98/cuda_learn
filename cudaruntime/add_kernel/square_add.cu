// square_add.cu
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// ✅ CUDA kernel
template <typename scalar_t>
__global__ void square_add_kernel(
    const scalar_t* a,
    const scalar_t* b,
    scalar_t* c,
    int64_t n) {

  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  if (idx < 5) {
    printf("Thread %ld: a[%ld] = %f, b[%ld] = %f\n",
           idx, idx, (double)a[idx], idx, (double)b[idx]);
  }

  c[idx] = a[idx] * a[idx] + b[idx] * b[idx];
}

// ✅ CUDA 实现（host 函数）
void square_add_cuda(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c) {
  int64_t n = a.numel();
  constexpr int threads_per_block = 256;
  int blocks = (n + threads_per_block - 1) / threads_per_block;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "square_add_cuda", ([&] {
    square_add_kernel<scalar_t><<<blocks, threads_per_block>>>(
        a.data_ptr<scalar_t>(),
        b.data_ptr<scalar_t>(),
        c.data_ptr<scalar_t>(),
        n
    );
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
  }
}

// ✅ CPU fallback（可选）
void square_add_cpu(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c) {
  auto a_data = a.data_ptr<float>();
  auto b_data = b.data_ptr<float>();
  auto c_data = c.data_ptr<float>();
  int64_t n = a.numel();

  for (int64_t i = 0; i < n; ++i) {
    c_data[i] = a_data[i] * a_data[i] + b_data[i] * b_data[i];
  }
}

// ✅ ✅ ✅ 关键：这才是 Python 能调用的入口函数！
torch::Tensor square_add(const torch::Tensor& a, const torch::Tensor& b) {
  TORCH_CHECK(a.device() == b.device(), "Inputs must be on the same device");
  TORCH_CHECK(a.sizes() == b.sizes(), "Inputs must have the same shape");

  auto c = torch::empty_like(a);

  if (a.is_cuda()) {
    square_add_cuda(a, b, c);
  } else {
    square_add_cpu(a, b, c);
  }

  return c;
}

// ✅ ✅ ✅ 必须有这一行：告诉 PyTorch “这个函数要暴露给 Python”
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("square_add", &square_add, "Square and add two tensors (CUDA optimized)");
}