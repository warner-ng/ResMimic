SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PRIMARY_CONDA_ROOT="/home/warner/miniconda3"
FALLBACK_CONDA_ROOT="${SCRIPT_DIR}/thirdparty/miniconda3"
if [[ -f "${PRIMARY_CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    CONDA_ROOT="${PRIMARY_CONDA_ROOT}"
else
    CONDA_ROOT="${FALLBACK_CONDA_ROOT}"
fi

if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate rl-motion
else
    source "${CONDA_ROOT}/bin/activate" rl-motion
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${CONDA_ROOT}/envs/rl-motion/lib"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
