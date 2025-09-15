# setup.py
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

setup(
    name='square_add',
    ext_modules=[
        CUDAExtension(
            name='square_add',          # ← 模块名，必须和 PYBIND11_MODULE 名字一致
            sources=['square_add.cu'],  # ← 只需要 .cu，PyTorch 自动处理绑定
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2', '--expt-relaxed-constexpr']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)