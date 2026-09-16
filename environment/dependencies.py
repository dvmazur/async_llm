"""Installation-time dependency check with one verified ARM wheel metadata defect."""
import importlib.metadata
import platform
import struct
import subprocess
import sys
import warnings


def check(uv):
    result = subprocess.run([uv, 'pip', 'check', '--python', sys.executable], text=True, capture_output=True)
    output = result.stdout + result.stderr
    print(output, end='')
    if not result.returncode:
        return
    known = 'The package `nvidia-cusparselt-cu13` was built for a different platform'
    issues = [line for line in output.splitlines() if line.startswith('The package ')]
    if platform.machine() != 'aarch64' or issues != [known] or 'Found 1 incompatibility' not in output:
        raise RuntimeError('dependency validation failed')
    dist = importlib.metadata.distribution('nvidia-cusparselt-cu13')
    if dist.version != '0.8.1' or 'Tag: py3-none-manylinux2014_sbsa' not in dist.read_text('WHEEL'):
        raise RuntimeError('unrecognized wheel metadata; do not suppress')
    library = dist.locate_file('nvidia/cusparselt/lib/libcusparseLt.so.0')
    with open(library, 'rb') as f:
        header = f.read(20)
    if header[:6] != b'\x7fELF\x02\x01' or struct.unpack('<H', header[18:20])[0] != 183:
        raise RuntimeError('cuSPARSELt binary is not ELF64 little-endian AArch64')
    warnings.warn('Locked cuSPARSELt 0.8.1 aarch64 wheel labels WHEEL as sbsa; '
                  'verified AArch64 ELF. This metadata-only exception does not waive other dependency checks.')


if __name__ == '__main__':
    check(sys.argv[1])
