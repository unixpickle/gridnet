from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="gridnet",
    ext_modules=[
        CUDAExtension(
            "gridnet_cuda",
            [
                "src/gridnet_cuda.cpp",
                "src/gridnet_cuda_kernels.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
