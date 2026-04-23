#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
SIM_DIR=""

usage() {
  cat <<USAGE
Usage:
  $0 --sim-dir <simulation_output_dir>

Behavior:
  - Renders dk diagnostics into <sim-dir>/processed/dk
  - Renders particle trace into <sim-dir>/processed/dk/particle_trace.png
  - Renders available E/B field frames into <sim-dir>/processed/fields/{E,B}
  - Renders available density center-slice frames into <sim-dir>/processed/density
  - Renders available mean-velocity vector frames into <sim-dir>/processed/velocity
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --sim-dir)
      if [ "$#" -lt 2 ]; then
        echo "Error: --sim-dir requires a value" >&2
        exit 1
      fi
      SIM_DIR=$2
      shift 2
      ;;
    --sim-dir=*)
      SIM_DIR=${1#--sim-dir=}
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Error: unknown argument '$1'" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ "$#" -gt 0 ]; then
  echo "Error: unexpected trailing arguments: $*" >&2
  usage >&2
  exit 1
fi

if [ -z "$SIM_DIR" ]; then
  echo "Error: --sim-dir is required" >&2
  usage >&2
  exit 1
fi

"$PYTHON_BIN" "$SCRIPT_DIR/plot_field_planes_3x3.py" --sim-dir "$SIM_DIR"
#"$PYTHON_BIN" "$SCRIPT_DIR/plot_dw_linear_fit.py" --sim-dir "$SIM_DIR"
#"$PYTHON_BIN" "$SCRIPT_DIR/plot_energy_component_sums.py" --sim-dir "$SIM_DIR"
#"$PYTHON_BIN" "$SCRIPT_DIR/plot_particle_trace.py" --sim-dir "$SIM_DIR"
"$PYTHON_BIN" "$SCRIPT_DIR/plot_density_planes_center.py" --sim-dir "$SIM_DIR"
"$PYTHON_BIN" "$SCRIPT_DIR/plot_mean_velocity_planes_center.py" --sim-dir "$SIM_DIR"
