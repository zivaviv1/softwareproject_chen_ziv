from setuptools import Extension, setup

module = Extension(
    "symnmfmodule",
    sources=["symnmfmodule.c", "symnmf.c"],
    libraries=["m"],
    extra_compile_args=['-Wall', '-Wextra', '-Werror', '-pedantic-errors']
)

setup(name="symnmfmodule", version="1.0", ext_modules=[module])