#!/usr/bin/env bash
set -euo pipefail

task_root="/home/Zhang_jinsong/spacebyte"
task_result="/data/students/Zhang_jinsong/spacebyte/result/pixel/spatialsense_pixel_baseline"
task_python="/data/students/Zhang_jinsong/spacebyte/envs/spacebyte-qformer/bin/python"

mkdir -p "${task_result}"
if [[ -f "${task_result}/run.pid" ]]; then
  task_previous_pid="$(<"${task_result}/run.pid")"
  if [[ "${task_previous_pid}" =~ ^[0-9]+$ ]] && kill -0 "${task_previous_pid}" 2>/dev/null; then
    echo "Pixel baseline is already running as PID ${task_previous_pid}" >&2
    exit 1
  fi
fi

cd "${task_root}"
nohup setsid "${task_python}" -u scripts/run_spatialsense_pixel_baseline_autonomous.py \
  --device cuda:2 \
  >"${task_result}/train.log" 2>&1 < /dev/null &
task_pid=$!
printf '%s\n' "${task_pid}" > "${task_result}/run.pid"
printf '%s\n' "${task_pid}"
