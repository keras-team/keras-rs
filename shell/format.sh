#!/bin/bash
set -Euo pipefail

base_dir=$(dirname $(dirname $0))
targets="${base_dir}"

ruff check --config "${base_dir}/pyproject.toml" --fix ${targets}
ruff format --config "${base_dir}/pyproject.toml" ${targets}
mypy --config-file "${base_dir}/pyproject.toml" ${targets}
