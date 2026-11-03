#!/bin/bash
# Create the `cable` conda environment and hook it up to the user-global Isaac Sim.
#
# After `conda activate cable`:
#   python -c "from isaacsim import SimulationApp"   # Isaac Sim 6.2 importable
#   python -m cablerouting ...                        # project code (editable install)
# Install extra packages with `cable-pip install ...` (pip with Isaac's paths hidden).
set -euo pipefail
cd "$(dirname "$0")/../.."
ENV_NAME=${ENV_NAME:-cable}
ISAACSIM_PATH=${ISAACSIM_PATH:-$HOME/opt/isaacsim/current}
[ -f "$ISAACSIM_PATH/setup_python_env.sh" ] || { echo "Isaac Sim not found at $ISAACSIM_PATH" >&2; exit 1; }

source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f environment.yml --prune
else
  conda env create -n "$ENV_NAME" -f environment.yml
fi
PREFIX=$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')

# Kit's interpreter must match ours exactly in major.minor.
kit_ver=$("$ISAACSIM_PATH/kit/python/bin/python3" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
env_ver=$("$PREFIX/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
[ "$kit_ver" = "$env_ver" ] || { echo "Python mismatch: Kit $kit_ver vs env $env_ver" >&2; exit 1; }

mkdir -p "$PREFIX/etc/conda/activate.d" "$PREFIX/etc/conda/deactivate.d"
cat > "$PREFIX/etc/conda/activate.d/isaacsim.sh" <<EOF
# Expose the user-global Isaac Sim install to this environment (written by create_cable_env.sh).
export _CABLE_SAVED_PYTHONPATH="\${PYTHONPATH-}"
export _CABLE_SAVED_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH-}"
export ISAACSIM_PATH="\${ISAACSIM_PATH:-$ISAACSIM_PATH}"
export CARB_APP_PATH="\$ISAACSIM_PATH/kit" ISAAC_PATH="\$ISAACSIM_PATH" EXP_PATH="\$ISAACSIM_PATH/apps"
source "\$ISAACSIM_PATH/setup_python_env.sh"
# This node registers a system CUDA 12.4 runtime with ldconfig; prefer the env's
# CUDA 12.8 runtime (shipped with torch) so torch and Kit agree on libcudart.so.12.
_cr="\$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
[ -d "\$_cr" ] && export LD_LIBRARY_PATH="\$_cr:\$LD_LIBRARY_PATH"
unset _cr
# Kit's own standard library must not shadow this interpreter's.
PYTHONPATH="\$(printf '%s' "\$PYTHONPATH" | tr ':' '\n' | grep -v -x -e "\$ISAACSIM_PATH/kit/python/lib/python3.12" -e '' | paste -sd: -)"
export PYTHONPATH
# Isaac Sim end-user license: accepted by the user when installing Isaac Sim.
export OMNI_KIT_ACCEPT_EULA=YES
alias cable-pip='env -u PYTHONPATH -u LD_LIBRARY_PATH python -m pip'
EOF
cat > "$PREFIX/etc/conda/deactivate.d/isaacsim.sh" <<'EOF'
export PYTHONPATH="${_CABLE_SAVED_PYTHONPATH-}"
export LD_LIBRARY_PATH="${_CABLE_SAVED_LD_LIBRARY_PATH-}"
[ -n "$PYTHONPATH" ] || unset PYTHONPATH
[ -n "$LD_LIBRARY_PATH" ] || unset LD_LIBRARY_PATH
unset _CABLE_SAVED_PYTHONPATH _CABLE_SAVED_LD_LIBRARY_PATH CARB_APP_PATH ISAAC_PATH EXP_PATH OMNI_KIT_ACCEPT_EULA
unalias cable-pip 2>/dev/null || true
EOF

env -u PYTHONPATH -u LD_LIBRARY_PATH "$PREFIX/bin/python" -m pip install -e . --no-deps
echo "Environment '$ENV_NAME' ready at $PREFIX"
