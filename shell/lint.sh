#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))
targets="${base_dir}"

isort --sp "${base_dir}/pyproject.toml" --check ${targets}
black --config "${base_dir}/pyproject.toml" --check ${targets}
flake8 --config "${base_dir}/setup.cfg" ${targets}
mypy --config-file "${base_dir}/pyproject.toml" ${targets}
