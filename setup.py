from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='torchtomo_cuda',
      ext_modules=[cpp_extension.CppExtension('torchtomo_cpp', ['torchtomo.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
