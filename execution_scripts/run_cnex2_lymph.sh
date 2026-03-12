#!/usr/bin/env bash
# run_cnex2_lymph.sh
#
# Description:
#   Launch the CellNeighborEX v2 Step 6+ pipeline for two conditions (PBS & MS),
#   each with their own precomputed sc_ccisignal.h5ad and sp_ccisignal.h5ad.
#
# Usage:
#   bash run_cnex2_lymph.sh
#   # Or background with logs:
#   nohup bash run_cnex2_lymph.sh > cnex2_lymph.nohup.log 2>&1 &

set -euo pipefail

############################################
# CONFIG (edit these)
############################################
# Conda
CONDA_BASE="/usr/local/anaconda3/2024.10"
CONDA_ENV="CellNeighborEX-env"

# Python entrypoint
PY_SCRIPT="/home/Data_Drive_8TB/khb/python_scripts/main_lymph.py"

# Output
OUTDIR="/home/Data_Drive_8TB/khb/data/cnex2_run_lymph_test"

# Precomputed files — PBS
REF_SIG_PBS="/home/Data_Drive_8TB/khb/data/PBS/sc_ccisignal.h5ad"
DECONV_PBS="/home/Data_Drive_8TB/khb/data/PBS/sp_ccisignal.h5ad"

# Precomputed files — MS
REF_SIG_MS="/home/Data_Drive_8TB/khb/data/MS/sc_ccisignal.h5ad"
DECONV_MS="/home/Data_Drive_8TB/khb/data/MS/sp_ccisignal.h5ad"

# Threads (BLAS/OpenMP hints)
THREADS="${THREADS:-24}"

# Clustering
DO_CLUSTER="false"          # set true if you need Leiden clustering from proportions
CLUSTER_INFO="sample_keys"  # must exist in both Visium .obs when DO_CLUSTER=false

# ccigenes
LOG_FC="1.0"
SPECIES="mouse"              # or mouse/rat/None
N_PERM="1000"
PERM_USE_ZEROS="true"
CHI_USE_ZEROS="false"

# ccipairs
RIDGE_ALPHA="1.0"
SELF_INTERACTION="false"
PVAL_TERM="0.05"

############################################
# LOGGING
############################################

timestamp() { date '+%Y-%m-%d_%H-%M-%S'; }
mkdir -p "${OUTDIR}"
LOG_DIR="${OUTDIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_$(timestamp).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "===== $(timestamp) : START CellNeighborEX v2 (dual-condition) ====="

############################################
# CONDA SETUP
############################################
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
echo "[INFO] Activating conda env: ${CONDA_ENV}"
conda activate "${CONDA_ENV}"

############################################
# THREAD HINTS
############################################
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"

############################################
# PRE-RUN CHECKS
############################################
if [ ! -f "${PY_SCRIPT}" ]; then
  echo "[ERROR] main_lymph.py not found at: ${PY_SCRIPT}"
  exit 1
fi

for f in "${REF_SIG_PBS}" "${DECONV_PBS}" "${REF_SIG_MS}" "${DECONV_MS}"; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] Required file missing: $f"
    exit 1
  fi
done

echo "[INFO] Python  : $(command -v python)"
python --version

############################################
# BUILD ARGUMENTS
############################################
ARGS=(
  --outdir "${OUTDIR}"
  --ref_signatures_file_PBS "${REF_SIG_PBS}"
  --deconv_file_PBS "${DECONV_PBS}"
  --ref_signatures_file_MS  "${REF_SIG_MS}"
  --deconv_file_MS  "${DECONV_MS}"
  --cluster_info "${CLUSTER_INFO}"
  --log_fc "${LOG_FC}"
  --n_permutations "${N_PERM}"
  --alpha "${RIDGE_ALPHA}"
  --pval_term "${PVAL_TERM}"
)

# Species
if [ -n "${SPECIES}" ]; then
  ARGS+=(--species "${SPECIES}")
fi

# Flags
if [ "${DO_CLUSTER}" = "true" ]; then
  ARGS+=(--do_cluster)
fi
if [ "${PERM_USE_ZEROS}" = "true" ]; then
  ARGS+=(--perm_use_zeros)
fi
if [ "${CHI_USE_ZEROS}" = "true" ]; then
  ARGS+=(--chi_use_zeros)
fi
if [ "${SELF_INTERACTION}" = "true" ]; then
  ARGS+=(--self_interaction)
fi

############################################
# RUN
############################################

echo "[INFO] Launching dual-condition pipeline..."
set +e
python "${PY_SCRIPT}" "${ARGS[@]}"
RET=$?
set -e

echo "--------------------------------------------------------"
if [ ${RET} -eq 0 ]; then
  echo "[SUCCESS] Dual-condition pipeline completed."
else
  echo "[FAILURE] Pipeline exited with code: ${RET}"
fi

echo "Logs saved to: ${LOG_FILE}"
echo "===== $(timestamp) : END CellNeighborEX v2 (dual-condition) ====="
