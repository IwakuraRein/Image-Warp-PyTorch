from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="pytorch-image-warp",
    version = "0.0.1", 
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "ImageWarp",
            ["ImageWarp.cpp", "image_warp.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
    install_requires=[         
        'torch'
    ]
)