from setuptools import Extension, setup

module = Extension(
    "symnmfmodule",
    sources=["symnmfmodule.c", "symnmf.c"],
    libraries=["m"],
)

setup(name="symnmfmodule", version="1.0", ext_modules=[module])