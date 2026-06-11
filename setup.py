from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os

sources = glob.glob(os.path.join('csrc', '*.cpp')) + glob.glob(os.path.join('csrc', '*.cu'))

setup(
  name='fast_gemm',
  version='0.1',
  packages=find_packages(),
  ext_modules=[
    CUDAExtension(
      name='fast_gemm_cuda',
      sources=sources,
      extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-lineinfo']}
    )
  ],
  cmdclass={'build_ext': BuildExtension}
)