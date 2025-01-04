#!/bin/bash
set -Euo pipefail

base_dir=$(dirname $(dirname $0))
targets="${base_dir}"

ruff check --config "${base_dir}/pyproject.toml" ${targets}
exitcode=$?

ruff format --check --config "${base_dir}/pyproject.toml" ${targets}
exitcode=$(($exitcode + $?))

mypy --config-file "${base_dir}/pyproject.toml" ${targets}
exitcode=$(($exitcode + $?))

exit $exitcode
