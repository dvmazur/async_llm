#!/usr/bin/env bash
# Shared, overridable paths. Machine-specific settings belong in ignored .runtime/config.sh.
ROBOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -r "$ROBOT_ROOT/robot_sim/.runtime/config.sh" ]]; then
    source "$ROBOT_ROOT/robot_sim/.runtime/config.sh"
fi
ROBOT_LOGIN_HOME="${ROBOT_LOGIN_HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)}"
ROBOT_CONDA_EXE="${ROBOT_CONDA_EXE:-${CONDA_EXE:-}}"
if [[ -z "$ROBOT_CONDA_EXE" ]]; then
    ROBOT_CONDA_EXE="$(command -v conda || true)"
fi
if [[ -z "$ROBOT_CONDA_EXE" || ! -x "$ROBOT_CONDA_EXE" ]]; then
    for ROBOT_CANDIDATE in "$ROBOT_LOGIN_HOME/miniconda3/bin/conda" "$ROBOT_LOGIN_HOME/miniforge3/bin/conda" /opt/conda/bin/conda; do
        if [[ -x "$ROBOT_CANDIDATE" ]]; then ROBOT_CONDA_EXE="$ROBOT_CANDIDATE"; break; fi
    done
fi
if [[ -z "$ROBOT_CONDA_EXE" || ! -x "$ROBOT_CONDA_EXE" ]]; then
    if [[ -z "${ROBOT_LLM_PYTHON:-}" || ! -x "$ROBOT_LLM_PYTHON" ]]; then
        echo "Set ROBOT_LLM_PYTHON to a Python environment, or install Conda" >&2
        return 2
    fi
fi
ROBOT_CONDA_BASE=
if [[ -n "$ROBOT_CONDA_EXE" && -x "$ROBOT_CONDA_EXE" ]]; then
    ROBOT_CONDA_BASE="$(cd "$(dirname "$ROBOT_CONDA_EXE")/.." && pwd)"
fi
if [[ -z "${ROBOT_ROS_ENV:-}" ]]; then
    if [[ -d "$ROBOT_ROOT/robot_sim/.runtime/ros/conda-meta" ]]; then
        ROBOT_ROS_ENV="$ROBOT_ROOT/robot_sim/.runtime/ros"
    else
        ROBOT_ROS_ENV="${ROBOT_CONDA_ENV:-double_p}"
    fi
fi
ROBOT_LLM_ENV="${ROBOT_LLM_ENV:-${ROBOT_CONDA_ENV:-async_llm}}"
ROBOT_LLM_PYTHON="${ROBOT_LLM_PYTHON:-$ROBOT_CONDA_BASE/envs/$ROBOT_LLM_ENV/bin/python}"
ROBOT_HF_HOME="${ROBOT_HF_HOME:-${HF_HOME:-$ROBOT_LOGIN_HOME/.cache/huggingface}}"
export ROBOT_CONDA_EXE ROBOT_CONDA_BASE ROBOT_ROS_ENV ROBOT_LLM_PYTHON ROBOT_HF_HOME
