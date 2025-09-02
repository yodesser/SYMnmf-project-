from setuptools import Extension, setup


module = Extension(
    "mysymnmf",
    sources=["symnmfmodule.c", "symnmf.c"],
    
    libraries=["m"],
)

setup(
    name="mysymnmf",
    version="1.1",
    description="Symmetric NMF (C extension)",
    ext_modules=[module],
)