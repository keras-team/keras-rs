#!/bin/bash

# ==============================================================================
# Script Configuration & Argument Handling
# ==============================================================================
# This script accepts up to three optional arguments:
# 1. Accelerator Type (default: v6e-8, options: v6e-8, v6e-16)
# 2. Zone (default: us-east5-a)
# 3. Project (default: tpu-prod-env-one-vm)

ACCELERATOR_TYPE=${1:-"v6e-8"}
ZONE=${2:-"us-east5-a"}
PROJECT=${3:-"tpu-prod-env-one-vm"}

# Validate the provided accelerator type
if [[ "${ACCELERATOR_TYPE}" != "v6e-8" && "${ACCELERATOR_TYPE}" != "v6e-16" ]]; then
    echo "Error: Invalid accelerator type '${ACCELERATOR_TYPE}'." >&2
    echo "Usage: $0 [v6e-8|v6e-16] [gcp_zone] [gcp_project]" >&2
    exit 1
fi

# ==============================================================================
# Environment Variables
# ==============================================================================
export TPU_NAME="abheesht-mlperf-${ACCELERATOR_TYPE}"
export ZONE
export PROJECT

echo ">>> Using Configuration:"
echo "    Accelerator: ${ACCELERATOR_TYPE}"
echo "    TPU Name:    ${TPU_NAME}"
echo "    Zone:        ${ZONE}"
echo "    Project:     ${PROJECT}"
echo "--------------------------------------------------"


# ==============================================================================
# TPU VM Creation
# ==============================================================================
echo ">>> Creating TPU VM: ${TPU_NAME} with accelerator ${ACCELERATOR_TYPE}..."
gcloud alpha compute tpus tpu-vm create ${TPU_NAME} \
  --zone=${ZONE} \
  --accelerator-type=${ACCELERATOR_TYPE} \
  --version=v2-alpha-tpuv6e \
  --project=${PROJECT} \
  --metadata=enable-oslogin=TRUE \
  --scopes=https://www.googleapis.com/auth/cloud-platform


# ==============================================================================
# Setup Python Virtual Environment on all workers
# ==============================================================================
echo ">>> Creating Python virtual environment..."
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
  --project ${PROJECT} \
  --zone ${ZONE} \
  --worker=all \
  --command="sudo apt-get update && sudo apt install -y python3.10-venv && python3 -m venv .keras-env"


# ==============================================================================
# Clone KerasRS and Install Dependencies
# ==============================================================================
echo ">>> Cloning KerasRS and installing dependencies..."
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
  --project ${PROJECT} \
  --zone ${ZONE} \
  --worker=all \
  --command="source .keras-env/bin/activate && git clone https://github.com/abheesht17/keras-rs.git && cd keras-rs && git checkout ml-perf && pip install -e . && pip install tensorflow-datasets && pip uninstall -y tensorflow keras && pip install git+https://github.com/keras-team/keras.git && pip install jax-tpu-embedding tensorflow-cpu"


# ==============================================================================
# Install TPU-compatible JAX
# ==============================================================================
echo ">>> Re-installing JAX for TPU compatibility..."
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
  --project ${PROJECT} \
  --zone ${ZONE} \
  --worker=all \
  --command="source .keras-env/bin/activate && pip uninstall -y jax jaxlib && pip install -U 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"


# ==============================================================================
# Verify Installation
# ==============================================================================
echo ">>> Verifying JAX installation..."
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
  --project ${PROJECT} \
  --zone ${ZONE} \
  --worker=all \
  --command="source .keras-env/bin/activate && echo 'import jax; print(jax.devices())' > script.py && python script.py"


# ==============================================================================
# Run Training Script
# ==============================================================================
echo ">>> Running the main script with config for ${ACCELERATOR_TYPE}..."
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
  --project ${PROJECT} \
  --zone ${ZONE} \
  --worker=all \
  --command="source .keras-env/bin/activate && cd keras-rs && python3 -m examples.ml_perf.main --config_name ${ACCELERATOR_TYPE}"

echo ">>> Script finished."
