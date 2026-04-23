#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLUMNS = ("t_[1/wpe]", "x_[c/wpe]", "y_[c/wpe]", "z_[c/wpe]")
DEFAULT_OUTPUT_NAME = "particle_trace.png"
DEFAULT_INPUT_NAMES = ("particle_trace.txt", "paricle_trace.txt")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Plot particle trace as 3D, XZ and XY trajectories."
  )
  parser.add_argument(
    "--sim-dir",
    type=Path,
    required=True,
    help="Simulation output directory with temporal/particle_trace.txt",
  )
  parser.add_argument(
    "--input",
    type=Path,
    default=None,
    help="Path to input particle_trace.txt (overrides --sim-dir)",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Path to output PNG (default: <sim-dir>/processed/dk/particle_trace.png)",
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


def parse_scalar(value: object, scales: dict[str, float]) -> float:
  if isinstance(value, (int, float)):
    return float(value)

  text = str(value).strip()
  match = re.fullmatch(
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(?:\[(.+)\])?",
    text,
  )
  if not match:
    raise ValueError(f"Unsupported scalar format: {value!r}")

  number = float(match.group(1))
  unit = match.group(2)
  if unit is None:
    return number

  unit = unit.strip()
  if unit not in scales:
    raise ValueError(f"Unsupported unit '[{unit}]' in value: {value!r}")
  return number * scales[unit]


def resolve_paths(args: argparse.Namespace) -> tuple[Path | None, Path, Path]:
  sim_dir = args.sim_dir.resolve()

  if args.input is not None:
    input_path = args.input.resolve()
  else:
    input_path = None
    for name in DEFAULT_INPUT_NAMES:
      candidate = sim_dir / "temporal" / name
      if candidate.exists():
        input_path = candidate
        break

  if args.output is not None:
    output_path = args.output.resolve()
  else:
    output_path = sim_dir / "processed" / "dk" / DEFAULT_OUTPUT_NAME

  config_path = sim_dir / "config.json"
  return input_path, output_path, config_path


def read_trace(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  with path.open("r", encoding="utf-8") as file:
    header_line = file.readline().strip()
    data_lines = file.readlines()

  header = re.split(r"\s{2,}", header_line) if header_line else []
  if not header:
    raise ValueError(f"Input file is empty or has no header: {path}")

  missing = [name for name in REQUIRED_COLUMNS if name not in header]
  if missing:
    raise ValueError("Missing required columns in header: " + ", ".join(missing))

  rows: list[list[float]] = []
  expected_columns = len(header)
  for line_no, line in enumerate(data_lines, start=2):
    stripped = line.strip()
    if not stripped:
      continue

    parts = stripped.split()
    if len(parts) != expected_columns:
      print(
        f"Warning: skipping malformed particle-trace row {line_no} in {path.name}: "
        f"expected {expected_columns} columns, got {len(parts)}",
        file=sys.stderr,
      )
      continue

    try:
      rows.append([float(value) for value in parts])
    except ValueError:
      print(
        f"Warning: skipping non-numeric particle-trace row {line_no} in {path.name}",
        file=sys.stderr,
      )

  if not rows:
    raise ValueError(f"No valid data rows found in: {path}")

  data = np.asarray(rows, dtype=float)
  if data.ndim == 1:
    data = data.reshape(1, -1)

  column_index = {name: idx for idx, name in enumerate(header)}
  max_idx = max(column_index[name] for name in REQUIRED_COLUMNS)
  if data.shape[1] <= max_idx:
    raise ValueError(
      "Data columns count is smaller than header-required indexes: "
      f"shape={data.shape}, max_required_index={max_idx}"
    )

  time = data[:, column_index["t_[1/wpe]"]]
  x = data[:, column_index["x_[c/wpe]"]]
  y = data[:, column_index["y_[c/wpe]"]]
  z = data[:, column_index["z_[c/wpe]"]]
  return time, x, y, z


def padded_limits(values: np.ndarray) -> tuple[float, float, float]:
  finite_values = np.asarray(values, dtype=float)
  finite_values = finite_values[np.isfinite(finite_values)]
  if finite_values.size == 0:
    return 0.0, 1.0, 1.0

  vmin = float(np.min(finite_values))
  vmax = float(np.max(finite_values))
  span = vmax - vmin
  if span <= 0.0:
    span = max(abs(vmin), abs(vmax), 1.0) * 1.0e-3

  pad = 0.05 * span
  padded_min = vmin - pad
  padded_max = vmax + pad
  padded_span = padded_max - padded_min
  return padded_min, padded_max, padded_span


def read_domain_limits(config_path: Path) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
  if not config_path.exists():
    return None

  with config_path.open("r", encoding="utf-8") as file:
    config = json.load(file)

  geometry = config["Geometry"]
  dx = parse_scalar(geometry["dx"], {})
  dy = parse_scalar(geometry["dy"], {})
  dz = parse_scalar(geometry["dz"], {})

  scales = {
    "dx": dx,
    "dy": dy,
    "dz": dz,
    "c/w_pe": 1.0,
    "1/w_pe": 1.0,
    "c/wpe": 1.0,
    "1/wpe": 1.0,
  }

  lx = parse_scalar(geometry["x"], scales)
  ly = parse_scalar(geometry["y"], scales)
  lz = parse_scalar(geometry["z"], scales)
  return (0.0, lx), (0.0, ly), (0.0, lz)


def limits_with_span(bounds: tuple[float, float]) -> tuple[float, float, float]:
  lower, upper = bounds
  span = upper - lower
  if span <= 0.0:
    span = 1.0
  return lower, upper, span


def apply_2d_limits(ax: plt.Axes, x_bounds: tuple[float, float], y_bounds: tuple[float, float]) -> None:
  ax.set_xlim(*x_bounds)
  ax.set_ylim(*y_bounds)
  if hasattr(ax, "set_box_aspect"):
    ax.set_box_aspect(1.0)



def build_plot(
  time: np.ndarray,
  x: np.ndarray,
  y: np.ndarray,
  z: np.ndarray,
  domain_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None,
) -> plt.Figure:
  fig = plt.figure(figsize=(16, 5.5), constrained_layout=True)
  ax_3d = fig.add_subplot(1, 3, 1, projection="3d")
  ax_xz = fig.add_subplot(1, 3, 2)
  ax_xy = fig.add_subplot(1, 3, 3)

  if domain_limits is None:
    x_bounds = padded_limits(x)[:2]
    y_bounds = padded_limits(y)[:2]
    z_bounds = padded_limits(z)[:2]
  else:
    x_bounds, y_bounds, z_bounds = domain_limits

  xmin, xmax, xspan = limits_with_span(x_bounds)
  ymin, ymax, yspan = limits_with_span(y_bounds)
  zmin, zmax, zspan = limits_with_span(z_bounds)

  line_kwargs = {
    "color": "tab:blue",
    "linewidth": 1.8,
  }
  start_kwargs = {
    "color": "tab:green",
    "s": 36,
    "label": "start",
    "zorder": 3,
  }
  end_kwargs = {
    "color": "tab:red",
    "s": 36,
    "label": "end",
    "zorder": 3,
  }

  ax_3d.plot(x, y, z, label="trajectory", **line_kwargs)
  ax_3d.scatter(x[0], y[0], z[0], **start_kwargs)
  ax_3d.scatter(x[-1], y[-1], z[-1], **end_kwargs)
  ax_3d.set_xlim(xmin, xmax)
  ax_3d.set_ylim(ymin, ymax)
  ax_3d.set_zlim(zmin, zmax)
  ax_3d.set_xlabel("x [c/w_pe]")
  ax_3d.set_ylabel("y [c/w_pe]")
  ax_3d.set_zlabel("z [c/w_pe]")
  ax_3d.set_title("3D trajectory")
  ax_3d.view_init(elev=20.0, azim=-55.0)
  if hasattr(ax_3d, "set_box_aspect"):
    ax_3d.set_box_aspect((1.0, 1.0, 1.0))
  ax_3d.legend(loc="best")

  ax_xz.plot(x, z, **line_kwargs)
  ax_xz.scatter(x[0], z[0], **start_kwargs)
  ax_xz.scatter(x[-1], z[-1], **end_kwargs)
  apply_2d_limits(ax_xz, x_bounds, z_bounds)
  ax_xz.set_xlabel("x [c/w_pe]")
  ax_xz.set_ylabel("z [c/w_pe]")
  ax_xz.set_title("XZ projection")
  ax_xz.grid(True, alpha=0.3)

  ax_xy.plot(x, y, **line_kwargs)
  ax_xy.scatter(x[0], y[0], **start_kwargs)
  ax_xy.scatter(x[-1], y[-1], **end_kwargs)
  apply_2d_limits(ax_xy, x_bounds, y_bounds)
  ax_xy.set_xlabel("x [c/w_pe]")
  ax_xy.set_ylabel("y [c/w_pe]")
  ax_xy.set_title("XY projection")
  ax_xy.grid(True, alpha=0.3)

  fig.suptitle(
    f"Particle trace, samples={time.size}, t=[{time[0]:.6g}, {time[-1]:.6g}] [1/w_pe]",
    fontsize=12,
  )
  return fig


def main() -> int:
  args = parse_args()

  try:
    input_path, output_path, config_path = resolve_paths(args)
    if input_path is None or not input_path.exists():
      print(
        f"Warning: particle trace input not found in '{args.sim_dir.resolve() / 'temporal'}'. "
        "Skipping particle trace rendering."
      )
      return 0

    time, x, y, z = read_trace(input_path)
    domain_limits = read_domain_limits(config_path)
    if domain_limits is None:
      print(f"Warning: config file does not exist: {config_path}. Using trace bounds for axes.")
    figure = build_plot(time, x, y, z, domain_limits)

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
