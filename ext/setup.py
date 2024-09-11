from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
setup(
    name='wn_treecode',
    packages=['wn_treecode'],
    ext_modules=[
        CUDAExtension('wn_treecode._cuda', [
            'wn_treecode/wn_treecode_cuda/wn_treecode_cuda_torch_interface.cu',
            'wn_treecode/wn_treecode_cuda/wn_treecode_cuda_kernels.cu',
        ],
        extra_compile_args={'cxx': ['-O3'],
                            'nvcc': ['-O3']}),

        CppExtension('wn_treecode._cpu', [
            'wn_treecode/wn_treecode_cpu/wn_treecode_cpu_torch_interface.cpp',
            'wn_treecode/wn_treecode_cpu/wn_treecode_cpu_treeutils.cpp',
            'wn_treecode/wn_treecode_cpu/wn_treecode_cpu_kernels.cpp',
        ],
        extra_compile_args={'cxx': ['-O3', '-fopenmp']}),

        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)