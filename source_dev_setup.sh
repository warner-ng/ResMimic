SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/thirdparty/miniconda3/bin/activate rl-motion
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SCRIPT_DIR}/thirdparty/miniconda3/envs/rl-motion/lib
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH