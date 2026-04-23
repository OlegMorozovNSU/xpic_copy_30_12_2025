#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLUMNS = ("Time", "dE+dB+dK")
DEFAULT_LINEAR_FILTER_THRESHOLD = 1.0e-3


def parse_args() -> argparse.Namespace:
  script_dir = Path(__file__).resolve().parent
  default_sim_dir = (script_dir / ".." / "output" / "drift_kinetic_ex3").resolve()

  parser = argparse.ArgumentParser(
    description="Plot dW points with linear fit through origin."
  )
  parser.add_argument(
    "--sim-dir",
    type=Path,
    default=default_sim_dir,
    help="Simulation output directory with temporal/dk_diagnostic.txt",
  )
  parser.add_argument(
    "--input",
    type=Path,
    default=None,
    help="Path to input dk_diagnostic.txt (overrides --sim-dir)",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Path to output PNG (default: <sim-dir>/processed/dk/dk_diagnostic_dw_linear_fit.png)",
  )
  parser.add_argument(
    "--config",
    type=Path,
    default=None,
    help="Path to config.json (default: <sim-dir>/config.json)",
  )
  parser.add_argument(
    "--linear-filter-threshold",
    type=float,
    default=DEFAULT_LINEAR_FILTER_THRESHOLD,
    help="Use only points with |dW| <= threshold for fit",
  )
  parser.add_argument(
    "--dpi",
    type=int,
    default=150,
    help="Output image DPI",
  )
  parser.add_argument(
    "--show",
    action="store_true",
    help="Show plot window after saving",
  )
  return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
  sim_dir = args.sim_dir.resolve()

  input_path = args.input.resolve() if args.input is not None else (sim_dir / "temporal" / "dk_diagnostic.txt")
  if args.output is not None:
    output_path = args.output.resolve()
  else:
    inferred_sim_dir = input_path.parent.parent
    output_path = inferred_sim_dir / "processed" / "dk" / "dk_diagnostic_dw_linear_fit.png"

  if args.config is not None:
    config_path = args.config.resolve()
  else:
    inferred_sim_dir = input_path.parent.parent
    config_path = inferred_sim_dir / "config.json"

  return input_path, output_path, config_path


def read_diagnostic(path: Path) -> tuple[np.ndarray, np.ndarray]:
  if not path.exists():
    raise FileNotFoundError(f"Input file does not exist: {path}")

  with path.open("r", encoding="utf-8") as file:
    header_line = file.readline().strip()

  header = re.split(r"\s{2,}", header_line) if header_line else []
  if not header:
    raise ValueError(f"Input file is empty or has no header: {path}")

  missing = [name for name in REQUIRED_COLUMNS if name not in header]
  if missing:
    raise ValueError("Missing required columns in header: " + ", ".join(missing))

  data = np.loadtxt(path, skiprows=1)
  if data.size == 0:
    raise ValueError(f"No data rows found in: {path}")
  if data.ndim == 1:
    data = data.reshape(1, -1)

  column_index = {name: i for i, name in enumerate(header)}
  max_idx = max(column_index[name] for name in REQUIRED_COLUMNS)
  if data.shape[1] <= max_idx:
    raise ValueError(
      "Data columns count is smaller than header-required indexes: "
      f"shape={data.shape}, max_required_index={max_idx}"
    )

  time_idx = data[:, column_index["Time"]]
  d_w = data[:, column_index["dE+dB+dK"]]
  return time_idx, d_w


def read_dt(config_path: Path) -> float:
  if not config_path.exists():
    raise FileNotFoundError(f"Config file does not exist: {config_path}")

  with config_path.open("r", encoding="utf-8") as file:
    config = json.load(file)

  dt = float(config["Geometry"]["dt"])
  if dt <= 0.0:
    raise ValueError(f"Geometry.dt must be positive, got: {dt}")
  return dt


def set_x_limits_to_available_data(ax: plt.Axes, time: np.ndarray) -> None:
  finite_time = np.asarray(time, dtype=float)
  finite_time = finite_time[np.isfinite(finite_time)]
  if finite_time.size == 0:
    ax.set_xlim(0.0, 1.0)
    return

  xmax = float(np.max(finite_time))
  if xmax <= 0.0:
    xmax = 1.0
  ax.set_xlim(0.0, xmax)


def build_plot(
  time: np.ndarray,
  d_w: np.ndarray,
  dt: float,
  linear_filter_threshold: float,
) -> plt.Figure:
  fig, ax = plt.subplots(figsize=(7, 7))

  mask_filtered = (
    np.isfinite(time)
    & np.isfinite(d_w)
    & (np.abs(d_w) <= linear_filter_threshold)
  )
  time_filtered = time[mask_filtered]
  d_w_filtered = d_w[mask_filtered]

  if time_filtered.size > 0:
    ax.plot(
      time_filtered,
      d_w_filtered,
      color="tab:blue",
      linestyle="-",
      linewidth=1.2,
      marker="o",
      markersize=3.0,
      label=rf"$|\Delta W|$",
    )
  else:
    ax.plot([], [], color="tab:blue", marker="o", linestyle="", label="No points after filter")

  if time_filtered.size >= 2:
    denom = float(np.dot(time_filtered, time_filtered))
    if denom > 0.0:
      # Fit y = kx so the line starts exactly at (0, 0).
      slope = float(np.dot(time_filtered, d_w_filtered) / denom)
      drift_per_dt = slope * dt
      fit_values = slope * time
      ax.plot(
        time,
        fit_values,
        color="tab:orange",
        linewidth=1.8,
        linestyle="--",
        label=rf"Linear fit, drift per $dt$: {drift_per_dt:.3e}",
      )
    else:
      ax.plot([], [], color="tab:orange", linestyle="--", label="Linear fit unavailable")
  else:
    ax.plot([], [], color="tab:orange", linestyle="--", label="Linear fit unavailable")

  set_x_limits_to_available_data(ax, time)
  ax.set_xlabel(r"Time, $\omega_p t$")
  ax.set_ylabel(r"$\Delta W$")
  ax.set_title(
    r"$\Delta W$, "
    + rf"$dt = {dt:g}\,[1/\omega_p]$"
  )
  ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.45)
  ax.grid(True, which="both", alpha=0.35)
  ax.legend(loc="best")

  if hasattr(ax, "set_box_aspect"):
    ax.set_box_aspect(1.0)

  fig.tight_layout()
  return fig


def main() -> int:
  args = parse_args()

  try:
    input_path, output_path, config_path = resolve_paths(args)
    time_idx, d_w = read_diagnostic(input_path)
    dt = read_dt(config_path)
    time = time_idx * dt

    figure = build_plot(
      time=time,
      d_w=d_w,
      dt=dt,
      linear_filter_threshold=args.linear_filter_threshold,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=args.dpi)
    print(f"Saved: {output_path}")

    if args.show:
      plt.show()

    plt.close(figure)
    return 0
  except Exception as exc:
    print(f"Error: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
