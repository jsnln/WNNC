from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
setup(
    name='torch_treecode',
    ext_modules=[
        CUDAExtension('torch_treecode_cuda', [
            'torch_treecode_cuda/torch_treecode_cuda.cpp',
            'torch_treecode_cuda/torch_treecode_cuda_kernels.cu',
        ],
        extra_compile_args={'cxx': ['-O3'],
                            'nvcc': ['-O3']}),

        CppExtension('torch_treecode_cpu', [
            'torch_treecode_cpu/torch_treecode_cpu.cpp',
            'torch_treecode_cpu/torch_treecode_cpu_kernels.cpp',
        ],
        extra_compile_args={'cxx': ['-O3', '-fopenmp']}),

        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)