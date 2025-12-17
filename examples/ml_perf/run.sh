#!/bin/bash
set -euo pipefail

# ==============================================================================
# Script Configuration & Argument Handling
# ==============================================================================
# This script accepts up to four optional arguments:
# 1. --accelerator-type (default: v6e-8, options: v6e-8, v6e-16)
# 2. --zone (default: us-east5-a)
# 3. --project (default: tpu-prod-env-one-vm)
# 4. --config-name (default: derived from accelerator type, e.g., v6e_8)

# Defaults
ACCELERATOR_TYPE="v6e-16"
ZONE="us-east5-a"
PROJECT="keras-team-gcp"
USER_CONFIG_NAME=""

# ==============================================================================
# Argument Parsing
# ==============================================================================

show_help() {
cat << EOF
Usage: $0 [--accelerator-type <type>] [--zone <zone>] [--project <project>] [--config-name <name>]
Options:
  --accelerator-type  The type of TPU accelerator (default: v6e-8). Options: v6e-8, v6e-16.
  --zone              The GCP zone for the TPU VM (default: us-east5-a).
  --project           The GCP project ID (default: tpu-prod-env-one-vm).
  --config-name       The specific configuration name to use for the training script.
                      (default: derived from accelerator type, e.g., v6e_8).
  -h, --help          Show this help message.
EOF
}


while [[ "$#" -gt 0 ]]; do
    case $1 in
        --accelerator-type) ACCELERATOR_TYPE="$2"; shift ;;
        --zone) ZONE="$2"; shift ;;
        --project) PROJECT="$2"; shift ;;
        --config-name) USER_CONFIG_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; show_help; exit 1 ;;
    esac
    shift
done

# Validate the provided accelerator type
if [[ "${ACCELERATOR_TYPE}" != "v6e-8" && "${ACCELERATOR_TYPE}" != "v6e-16" ]]; then
    echo "Error: Invalid accelerator type '${ACCELERATOR_TYPE}'." >&2
    show_help
    exit 1
fi

# ==============================================================================
# Environment Variables
# ==============================================================================
export TPU_NAME="${USER}-mlperf-${ACCELERATOR_TYPE}"
export ZONE
export PROJECT

# Use the user-provided config name if it exists, otherwise derive it.
if [[ -n "${USER_CONFIG_NAME}" ]]; then
  export CONFIG_NAME=${USER_CONFIG_NAME}
else
  export CONFIG_NAME=${ACCELERATOR_TYPE//-/_}
fi

echo ">>> Using Configuration:"
echo "    Accelerator: ${ACCELERATOR_TYPE}"
echo "    TPU Name:    ${TPU_NAME}"
echo "    Zone:        ${ZONE}"
echo "    Project:     ${PROJECT}"
echo "    Config Name: ${CONFIG_NAME}"
echo "--------------------------------------------------"


# ==============================================================================
# TPU VM Creation
# ==============================================================================
echo ">>> Checking for existing TPU VM: ${TPU_NAME}..."
if gcloud alpha compute tpus tpu-vm describe ${TPU_NAME} --zone=${ZONE} --project=${PROJECT} &> /dev/null; then
  echo ">>> TPU VM '${TPU_NAME}' already exists. Skipping creation."
else
  echo ">>> Creating TPU VM: ${TPU_NAME} with accelerator ${ACCELERATOR_TYPE}..."
  gcloud alpha compute tpus tpu-vm create ${TPU_NAME} \
    --zone=${ZONE} \
    --accelerator-type=${ACCELERATOR_TYPE} \
    --version=v2-alpha-tpuv6e \
    --project=${PROJECT} \
    --metadata=enable-oslogin=TRUE \
    --scopes=https://www.googleapis.com/auth/cloud-platform
fi


# ==============================================================================
# Setup Python venv on all workers
# ==============================================================================
echo ">>> Checking for Python virtual environment..."
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
  --project ${PROJECT} \
  --zone ${ZONE} \
  --worker=all \
  --command="sudo apt-get update && sudo apt install -y python3.12-venv && if [ ! -d '.keras-env' ]; then echo '>>> Creating .keras-env...'; python3.12 -m venv .keras-env; else echo '>>> .keras-env already exists.'; fi"


# ==============================================================================
# Clone/Update KerasRS and Install Dependencies
# ==============================================================================
echo ">>> Cloning or updating KerasRS and installing dependencies..."
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
  --project ${PROJECT} \
  --zone ${ZONE} \
  --worker=all \
  --command="
    set -e # Ensure script exits on error
    source .keras-env/bin/activate

    if [ ! -d 'keras-rs' ]; then
      echo '>>> Cloning keras-rs repository...'
      git clone https://github.com/keras-team/keras-rs.git
      cd keras-rs
    else
      echo '>>> keras-rs repository exists. Pulling latest changes...'
      cd keras-rs
      git pull
    fi

    echo '>>> Installing/updating dependencies...'
    pip install -e .
    pip install -U jax-tpu-embedding tensorflow-cpu keras
  "


# ==============================================================================
# Install TPU-compatible JAX
# ==============================================================================
echo ">>> Re-installing JAX for TPU compatibility..."
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
  --project ${PROJECT} \
  --zone ${ZONE} \
  --worker=all \
  --command="source .keras-env/bin/activate && pip install -U 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"

# ==============================================================================
# Kill Previous Training Processes
# ==============================================================================
# echo ">>> Listing matching processes..."
# gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
#   --project ${PROJECT} \
#   --zone ${ZONE} \
#   --worker=all \
#   --command="ps aux | grep '[e]xamples.ml_perf.main' || true"

# echo ">>> Terminating any existing training processes..."
# gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
#   --project ${PROJECT} \
#   --zone ${ZONE} \
#   --worker=all \
#   --command="pkill -9 -f 'python3.12 -m examples.ml_perf.[m]ain.*' || true"

# ==============================================================================
# Verify Installation
# ==============================================================================
echo ">>> Verifying JAX installation..."
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
  --project ${PROJECT} \
  --zone ${ZONE} \
  --worker=all \
  --command="source .keras-env/bin/activate && python3.12 -c 'import jax; print(jax.devices())'"


# ==============================================================================
# Run Training Script
# ==============================================================================
echo ">>> Running the main script with config for ${ACCELERATOR_TYPE}..."
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
  --project ${PROJECT} \
  --zone ${ZONE} \
  --worker=all \
  --command="source .keras-env/bin/activate && cd keras-rs && python3.12 -m examples.ml_perf.main --config_name ${CONFIG_NAME}"

echo ">>> Script finished."
