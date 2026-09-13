#!/bin/bash
# Shared helper for a relocatable CPython under $DEPS_DIR.
set -euo pipefail

# Keep pip from satisfying deps from the host user site.
export PYTHONNOUSERSITE=1

# nemo-gym's pyproject.toml requires >=3.13.14, so a 3.12 runtime cannot install
# it: pip aborts with "Package 'nemo-gym' requires a different Python".
PYTHON_VERSION="${PYTHON_VERSION:-3.13.15}"
PBS_RELEASE="${PBS_RELEASE:-20260901}"

# Detect the host architecture instead of assuming x86_64. The runtime is built
# here but executed inside the task container, so a mismatch surfaces there as
# "cannot execute binary file: Exec format error".
detect_pbs_arch() {
    local machine
    machine="$(uname -m)"
    case "$machine" in
        x86_64 | amd64) echo "x86_64-unknown-linux-gnu" ;;
        aarch64 | arm64) echo "aarch64-unknown-linux-gnu" ;;
        *)
            echo "Unsupported architecture for python-build-standalone: $machine" >&2
            return 1
            ;;
    esac
}

ARCH="${ARCH:-$(detect_pbs_arch)}"

install_portable_python() {
    if [ -x "$DEPS_DIR/bin/python3" ]; then
        echo "Portable python already present at $DEPS_DIR/bin/python3"
        return 0
    fi
    local url="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_RELEASE}/cpython-${PYTHON_VERSION}+${PBS_RELEASE}-${ARCH}-install_only.tar.gz"
    echo "Downloading portable python: $url"
    # Tarball extracts to python/{bin,lib}.
    curl -fsSL "$url" | tar xz -C "$DEPS_DIR" --strip-components=1
    "$DEPS_DIR/bin/python3" -m pip install --upgrade pip
}

install_nemo_gym_deps() {
    echo "Installing NeMo-Gym deps from $NEMO_GYM_ROOT"
    "$DEPS_DIR/bin/python3" -m pip install "$NEMO_GYM_ROOT"
}
