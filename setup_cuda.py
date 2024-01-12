from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='torchtomo_cuda',
    ext_modules=[
        CUDAExtension('torchtomo_cuda', [
            'torchtomo.cpp',
            'torchtomo_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
