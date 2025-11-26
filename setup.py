from setuptools import setup, Extension
import pybind11
import subprocess
import os
import sys


def parse_pkg_config(packages):
    def run(args):
        return subprocess.check_output(args).decode().strip().split()

    cflags = run(["pkg-config", "--cflags"] + list(packages))
    libs = run(["pkg-config", "--libs"] + list(packages))

    include_dirs = []
    library_dirs = []
    libraries = []
    extra_compile_args = []
    extra_link_args = []

    for token in cflags:
        if token.startswith("-I"):
            include_dirs.append(token[2:])
        else:
            extra_compile_args.append(token)

    for token in libs:
        if token.startswith("-L"):
            library_dirs.append(token[2:])
        elif token.startswith("-l"):
            libraries.append(token[2:])
        else:
            extra_link_args.append(token)

    return include_dirs, library_dirs, libraries, extra_compile_args, extra_link_args


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
APPS_DIR = os.path.join(PROJECT_ROOT, "apps")
source_file = os.path.join(APPS_DIR, "rpicam_hello_py.cpp")

if not os.path.exists(source_file):
    sys.stderr.write(f"Could not find {source_file}\n")
    sys.exit(1)

# Use the same deps Meson uses: libcamera_dep + rpicam_app (via pkg-config)
pkg_inc, pkg_lib_dirs, pkg_libs, pkg_cflags, pkg_ldflags = parse_pkg_config(
    ["libcamera", "rpicam_app"]
)

include_dirs = [
    pybind11.get_include(),
    PROJECT_ROOT,   # so "core/..." works
] + pkg_inc

extra_compile_args = ["-O3", "-std=c++17"] + pkg_cflags
extra_link_args = pkg_ldflags

ext = Extension(
    "rpicam_hello",
    sources=[source_file],
    include_dirs=include_dirs,
    library_dirs=pkg_lib_dirs,
    libraries=pkg_libs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)

setup(
    name="rpicam_hello",
    version="0.0.1",
    description="pybind11 wrapper for rpicam-hello",
    ext_modules=[ext],
)



