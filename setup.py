import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = []
if torch.cuda.is_available():
    ext_modules.append(
        CUDAExtension(
            "gridnet_cuda",
            [
                "src/gridnet_cuda.cpp",
                "src/gridnet_cuda_kernels.cu",
            ],
        )
    )

setup(
    name="gridnet",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch",
        "pytest-benchmark",
    ],
)
