#!/usr/bin/env bash
# run_cnex2.sh
#
# Description:
#   Launch the end-to-end CellNeighborEX v2 pipeline (ccisignal → ccigenes → ccipairs)
#   using the updated main.py that imports from ccisignal, ccigenes, and ccipairs.
#
# Usage:
#   bash run_cnex2.sh
#
#   # Run in background with logs:
#   nohup bash run_cnex2.sh > cnex2_run.log 2>&1 &
#
# Notes:
#   - Edit the CONFIG section to match your paths and parameters.
#   - START_FROM=1 → run full pipeline (ccisignal → ccigenes → ccipairs).
#   - START_FROM=6 → resume from ccigenes with precomputed .h5ad files.
#   
#   - The following parameters must be customized for each dataset:
#
#  	• [ccisignal_module] LABEL_KEY  
#    	Column name in the scRNA-seq reference `.obs` that contains cell type annotations.  
#    	(e.g., "celltype", "predictions", or another user-defined label column)
#
# 	• [ccigenes_module] CLUSTER_INFO  
#    	Column name in the Visium `.obs` that contains spatial cluster labels.  
#    	This value differs across datasets (see table in README).
#
#  	• [ccigenes_module] SPECIES  
#    	Species of the dataset ("human", "mouse", or "rat").  
#    	Required for gene symbol simplification and database lookup (see table in README).
#
#
# =============================================================================
# Pipeline Resume Options – Details
# =============================================================================
# Stages
#   1) ccisignal   → builds reference signatures & deconvolves Visium
#   2) ccigenes    → infers cell–cell interaction (CCI) genes
#   3) ccipairs    → assigns interacting cell-type pairs for each CCI gene
#
# Key outputs
#   - [ccisignal_output] sc_ccisignal.h5ad: scRNA-seq reference with cell-type–specific expression
#   - [ccisignal_output]sp_ccisignal.h5ad: Visium with expected expression per spot/gene
#   - [ccigenes_output]ccigenes_results.csv: list of inferred CCI genes with statistics
#   - [ccipairs_output]ccipairs_results.csv: interacting cell-type pair(s) for each CCI gene
#
# START_FROM="1" (full run; steps 1–7)
#   - Generates sc_ccisignal.h5ad, sp_ccisignal.h5ad, and final CSVs.
#
# START_FROM="6" (resume)
#   - Skips ccisignal; runs only ccigenes → ccipairs.
#   - Requires:
#       REF_SIG_FILE=/path/to/sc_ccisignal.h5ad
#       DECONV_FILE=/path/to/sp_ccisignal.h5ad
#
# =============================================================================

set -euo pipefail

############################################
# CONFIG (edit these as needed)
############################################
# Conda
CONDA_BASE="/usr/local/anaconda3/2024.10"
CONDA_ENV="CellNeighborEX-env"

# Python entrypoint
PY_SCRIPT="/home/Data_Drive_8TB/khb/python_scripts/main.py"

# Inputs (used when START_FROM=1)
REF_H5AD="/home/Data_Drive_8TB/khb/data/sc_simulation.h5ad"         # annotated scRNA-seq reference
VISIUM_H5AD="/home/Data_Drive_8TB/khb/data/sp_simulation_cci.h5ad"  # Visium ST data

# Output root
OUTDIR="/home/Data_Drive_8TB/khb/data/cnex2_run_test"

# Threads (BLAS/OpenMP hints)
THREADS="${THREADS:-24}"

# ===== Resume-from-step6 options =====
# 1 = run full pipeline (Steps 1–7), 6 = resume from Step 6 with precomputed outputs
#START_FROM="${START_FROM:-1}"
#REF_SIG_FILE="${REF_SIG_FILE:-${OUTDIR}/reference_signatures/sc_ccisignal.h5ad}"
#DECONV_FILE="${DECONV_FILE:-${OUTDIR}/cell2location_map/sp_ccisignal.h5ad}"

START_FROM="6"
REF_SIG_FILE="/home/Data_Drive_8TB/khb/data/sc_ccisignal.h5ad" # file path where "sc_ccisignal.h5ad" is located
DECONV_FILE="/home/Data_Drive_8TB/khb/data/sp_ccisignal.h5ad" # file path where "sp_ccisignal.h5ad" is located


############################################
# Core analysis options (ccisignal module)
############################################
LABEL_KEY="predictions"        # [ccisignal] Column in scRNA-seq reference .obs containing cell type annotations (e.g. user-specified)
REF_BATCH_KEY=""               # [ccisignal] Optional batch key in scRNA-seq reference .obs; leave empty if no batch correction
REF_DO_FILTERING="false"       # [ccisignal] Whether to filter genes before training the RegressionModel
REF_MAX_EPOCHS="250"           # [ccisignal] Number of training epochs for the RegressionModel (reference signatures)

SP_BATCH_KEY="sample"	       # [ccisignal] Optional batch key in spatial transcriptomics .obs; leave empty if no batch correction
SP_MAX_EPOCHS="4000"           # [ccisignal] Number of training epochs for the Cell2location deconvolution model
SP_LR="0.002"                  # [ccisignal] Learning rate for the Cell2location deconvolution model
N_CELLS_PER_LOCATION="10"      # [ccisignal] Expected average number of cells per Visium spot
DETECTION_ALPHA="20"           # [ccisignal] Hyperparameter controlling normalization of RNA detection rates across spots

DO_CLUSTER="false" 		  # [ccisignal] Optional clustering; perform clustering based on spot proportions after deconvolution (true/false)

############################################
# Downstream options (ccigenes module)
############################################
CLUSTER_INFO="spatial_kmeans"    # [ccigenes] Cluster label column in Visium .obs (e.g. user-specified)
LOG_FC="0.5"                     # [ccigenes] Minimum log2 fold-change threshold for defining niche-specific genes
SPECIES="human"                  # [ccigenes] Species name (human/mouse/rat) for gene symbol simplification and DB lookup
N_PERM="1000"                    # [ccigenes] Number of permutations for the permutation test
PERM_USE_ZEROS="true"            # [ccigenes] Include zeros in permutation test?
CHI_USE_ZEROS="false"            # [ccigenes] Include zeros in chi-square test?

############################################
# Regression options (ccipairs module)
############################################
RIDGE_ALPHA="1.0"                # [ccipairs] Ridge (L2) regularization strength for the initial NB regression
SELF_INTERACTION="false"         # [ccipairs] Allow self × self interactions?
PVAL_TERM="0.05"                 # [ccipairs] p-value threshold for selecting significant interaction terms

############################################
# LOGGING
############################################
timestamp() { date '+%Y-%m-%d_%H-%M-%S'; }
mkdir -p "${OUTDIR}"
LOG_DIR="${OUTDIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_$(timestamp).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "===== $(timestamp) : START CellNeighborEX v2 pipeline ====="

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
# PRE-RUN CHECKS (mode-aware)
############################################
if [ ! -f "${PY_SCRIPT}" ]; then
  echo "[ERROR] main.py not found at: ${PY_SCRIPT}"
  exit 1
fi

if [ "${START_FROM}" = "6" ]; then
  # Resume mode: check precomputed outputs instead of raw inputs
  if [ ! -f "${REF_SIG_FILE}" ]; then
    echo "[ERROR] Precomputed ref signatures not found: ${REF_SIG_FILE}"
    echo "        Set REF_SIG_FILE or place the file at the default path."
    exit 1
  fi
  if [ ! -f "${DECONV_FILE}" ]; then
    echo "[ERROR] Precomputed deconvolution map not found: ${DECONV_FILE}"
    echo "        Set DECONV_FILE or place the file at the default path."
    exit 1
  fi
else
  # Full run: need raw input files
  if [ ! -f "${REF_H5AD}" ]; then
    echo "[ERROR] scRNA-seq h5ad not found: ${REF_H5AD}"
    exit 1
  fi
  if [ ! -f "${VISIUM_H5AD}" ]; then
    echo "[ERROR] Visium h5ad not found: ${VISIUM_H5AD}"
    exit 1
  fi
fi

echo "[INFO] Python     : $(command -v python)"
python --version
echo "[INFO] main.py    : ${PY_SCRIPT}"
echo "[INFO] OUTDIR     : ${OUTDIR}"
echo "[INFO] THREADS    : ${THREADS}"
echo "[INFO] START_FROM : ${START_FROM}"
if [ "${START_FROM}" = "6" ]; then
  echo "[INFO] REF_SIG_FILE : ${REF_SIG_FILE}"
  echo "[INFO] DECONV_FILE  : ${DECONV_FILE}"
else
  echo "[INFO] REF_H5AD     : ${REF_H5AD}"
  echo "[INFO] VISIUM_H5AD  : ${VISIUM_H5AD}"
fi
echo "--------------------------------------------------------"

############################################
# BUILD ARGUMENT LIST (mode-aware)
############################################
ARGS=()

if [ "${START_FROM}" = "6" ]; then
  # Resume from Step 6: no positional inputs; pass explicit files
  ARGS+=(--outdir "${OUTDIR}")
  ARGS+=(--start_from 6)
  ARGS+=(--ref_signatures_file "${REF_SIG_FILE}")
  ARGS+=(--deconv_file "${DECONV_FILE}")
else
  # Full run (Steps 1–7): positional inputs first
  ARGS+=("${REF_H5AD}")
  ARGS+=("${VISIUM_H5AD}")
  ARGS+=(--outdir "${OUTDIR}")
fi

# ccisignal args (common)
ARGS+=(--label_key "${LABEL_KEY}")
if [ -n "${REF_BATCH_KEY}" ]; then
  ARGS+=(--ref_batch_key "${REF_BATCH_KEY}")
fi
if [ "${REF_DO_FILTERING}" = "true" ]; then
  ARGS+=(--ref_do_filtering)
fi
ARGS+=(--ref_max_epochs "${REF_MAX_EPOCHS}")
ARGS+=(--sp_max_epochs "${SP_MAX_EPOCHS}")
ARGS+=(--sp_lr "${SP_LR}")
ARGS+=(--n_cells_per_location "${N_CELLS_PER_LOCATION}")
ARGS+=(--detection_alpha "${DETECTION_ALPHA}")

# Optional proportion-based clustering
if [ "${DO_CLUSTER}" = "true" ]; then
  ARGS+=(--do_cluster)
  ARGS+=(--cluster_info "${CLUSTER_INFO}")   # will produce 'proportion_leiden' by default, but keep explicit
else
  ARGS+=(--cluster_info "${CLUSTER_INFO}")   # use existing column name in Visium .obs
fi

# ccigenes args (common)
ARGS+=(--log_fc "${LOG_FC}")
if [ -n "${SPECIES}" ]; then
  ARGS+=(--species "${SPECIES}")
fi
ARGS+=(--n_permutations "${N_PERM}")
if [ "${PERM_USE_ZEROS}" = "true" ]; then
  ARGS+=(--perm_use_zeros)
fi
if [ "${CHI_USE_ZEROS}" = "true" ]; then
  ARGS+=(--chi_use_zeros)
fi

# ccipairs args (common)
ARGS+=(--alpha "${RIDGE_ALPHA}")
if [ "${SELF_INTERACTION}" = "true" ]; then
  ARGS+=(--self_interaction)
fi
ARGS+=(--pval_term "${PVAL_TERM}")

############################################
# RUN
############################################
echo "[INFO] Launching pipeline..."
set +e
python "${PY_SCRIPT}" "${ARGS[@]}"
RET=$?
set -e

echo "--------------------------------------------------------"
if [ ${RET} -eq 0 ]; then
  echo "[SUCCESS] Pipeline completed."
else
  echo "[FAILURE] Pipeline exited with code: ${RET}"
fi

echo "Logs saved to: ${LOG_FILE}"
echo "===== $(timestamp) : END CellNeighborEX v2 pipeline ====="
