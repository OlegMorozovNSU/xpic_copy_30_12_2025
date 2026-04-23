#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLANES = ("XY", "XZ", "YZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Render center XY/XZ/YZ slices for particle density diagnostics."
  )
  parser.add_argument(
    "--sim-dir",
    type=Path,
    required=True,
    help="Simulation output directory with config.json and <sort>/density/",
  )
  parser.add_argument(
    "--output-root",
    type=Path,
    default=None,
    help="Output directory for PNG frames (default: <sim-dir>/processed/density)",
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
    help="Show figures while rendering",
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


def load_config(
  config_path: Path,
) -> tuple[list[str], int, int, int, float, float, float, float, int]:
  with config_path.open("r", encoding="utf-8") as file:
    config = json.load(file)

  particles = config.get("Particles", [])
  sort_names = [str(item["sort_name"]) for item in particles if "sort_name" in item]
  if not sort_names:
    raise ValueError("No particles with sort_name found in config.json")

  geometry = config["Geometry"]
  dx = parse_scalar(geometry["dx"], {})
  dy = parse_scalar(geometry["dy"], {})
  dz = parse_scalar(geometry["dz"], {})
  dt = parse_scalar(geometry["dt"], {})

  scales = {
    "dx": dx,
    "dy": dy,
    "dz": dz,
    "dt": dt,
    "c/w_pe": 1.0,
    "1/w_pe": 1.0,
    "c/wpe": 1.0,
    "1/wpe": 1.0,
  }

  lx = parse_scalar(geometry["x"], scales)
  ly = parse_scalar(geometry["y"], scales)
  lz = parse_scalar(geometry["z"], scales)
  total_time = parse_scalar(geometry["t"], scales)
  diagnose_period = parse_scalar(geometry["diagnose_period"], scales)

  nx = int(round(lx / dx))
  ny = int(round(ly / dy))
  nz = int(round(lz / dz))
  time_width = len(str(int(round(total_time / dt))))

  return sort_names, nx, ny, nz, lx, ly, lz, diagnose_period, time_width


def list_snapshot_files(path: Path, *, name_width: int, size_bytes: int) -> list[Path]:
  files = [
    item for item in path.iterdir()
    if item.is_file()
    and item.name.isdigit()
    and len(item.name) == name_width
    and item.stat().st_size == size_bytes
  ]
  if not files:
    raise FileNotFoundError(f"No valid snapshot files found in '{path}'")

  def key(file_path: Path) -> tuple[int, str]:
    return int(file_path.name), file_path.name

  return sorted(files, key=key)


def nonzero_limits(vmin: float, vmax: float) -> tuple[float, float]:
  if vmin != vmax:
    return vmin, vmax
  eps = 1.0e-12 if vmin == 0.0 else abs(vmin) * 1.0e-6
  return vmin - eps, vmax + eps


def read_density_frame(path: Path, nx: int, ny: int, nz: int) -> np.ndarray:
  data = np.fromfile(path, dtype=np.float32)
  expected = nx * ny * nz
  if data.size != expected:
    raise ValueError(
      f"Unexpected size in '{path}': {data.size}, expected {expected} (nx={nx}, ny={ny}, nz={nz})"
    )
  return data.reshape((nz, ny, nx))


def get_common_snapshot_names(
  sort_dirs: dict[str, Path],
  *,
  name_width: int,
  size_bytes: int,
) -> list[str]:
  names_per_sort = {
    sort_name: {
      path.name for path in list_snapshot_files(
        sort_dir,
        name_width=name_width,
        size_bytes=size_bytes,
      )
    }
    for sort_name, sort_dir in sort_dirs.items()
  }

  common = None
  for names in names_per_sort.values():
    common = set(names) if common is None else common.intersection(names)

  if not common:
    raise FileNotFoundError("No common snapshot files among all particle sorts.")

  return sorted(common, key=lambda name: (int(name), name))


def compute_sort_limits(
  sort_dirs: dict[str, Path],
  names: list[str],
  nx: int,
  ny: int,
  nz: int,
) -> dict[str, tuple[float, float]]:
  limits: dict[str, tuple[float, float]] = {}

  for sort_name, sort_dir in sort_dirs.items():
    vmin = np.inf
    vmax = -np.inf

    for name in names:
      frame = read_density_frame(sort_dir / name, nx, ny, nz)
      vmin = min(vmin, float(np.min(frame)))
      vmax = max(vmax, float(np.max(frame)))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
      limits[sort_name] = (0.0, 1.0)
    else:
      limits[sort_name] = nonzero_limits(vmin, vmax)

  return limits


def clear_existing_pngs(path: Path) -> None:
  if not path.exists():
    return
  for png in path.glob("*.png"):
    png.unlink()


def render_frame(
  frame_idx: int,
  name: str,
  sort_names: list[str],
  sort_dirs: dict[str, Path],
  sort_limits: dict[str, tuple[float, float]],
  nx: int,
  ny: int,
  nz: int,
  lx: float,
  ly: float,
  lz: float,
  diagnose_period: float,
  out_dir: Path,
  dpi: int,
  show: bool,
) -> None:
  nsorts = len(sort_names)
  fig, axes = plt.subplots(
    nsorts,
    3,
    figsize=(15, max(4.2, 3.6 * nsorts)),
    constrained_layout=True,
  )

  if nsorts == 1:
    axes = np.asarray([axes])

  z_center = nz // 2
  y_center = ny // 2
  x_center = nx // 2

  for row, sort_name in enumerate(sort_names):
    frame = read_density_frame(sort_dirs[sort_name] / name, nx, ny, nz)

    slices = {
      "XY": frame[z_center, :, :],
      "XZ": frame[:, y_center, :],
      "YZ": frame[:, :, x_center],
    }

    extents = {
      "XY": (0.0, lx, 0.0, ly),
      "XZ": (0.0, lx, 0.0, lz),
      "YZ": (0.0, ly, 0.0, lz),
    }

    axis_labels = {
      "XY": ("x [c/w_pe]", "y [c/w_pe]"),
      "XZ": ("x [c/w_pe]", "z [c/w_pe]"),
      "YZ": ("y [c/w_pe]", "z [c/w_pe]"),
    }

    vmin, vmax = sort_limits[sort_name]
    row_mappable = None

    for col, plane in enumerate(PLANES):
      image = axes[row, col].imshow(
        slices[plane],
        origin="lower",
        aspect="auto",
        extent=extents[plane],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
      )
      row_mappable = image

      axes[row, col].set_title(f"{sort_name}: {plane}")
      axes[row, col].set_xlabel(axis_labels[plane][0])
      axes[row, col].set_ylabel(axis_labels[plane][1])
      axes[row, col].grid(False)

    colorbar = fig.colorbar(
      row_mappable,
      ax=axes[row, :],
      shrink=0.95,
      fraction=0.03,
      pad=0.02,
    )
    colorbar.set_label(f"{sort_name} density")

  time_value = frame_idx * diagnose_period
  fig.suptitle(
    (
      f"Density center slices, frame={frame_idx}, t={time_value:.6g} [1/w_pe], file={name}\n"
      f"centers: z={z_center}, y={y_center}, x={x_center}"
    ),
    fontsize=12,
  )

  out_path = out_dir / f"{name}.png"
  fig.savefig(out_path, dpi=dpi)
  if show:
    plt.show()
  plt.close(fig)

  print(f"Saved: {out_path} (frame={frame_idx}, t={time_value:.6g})")


def main() -> int:
  args = parse_args()

  try:
    sim_dir = args.sim_dir.resolve()
    config_path = sim_dir / "config.json"
    if not config_path.exists():
      raise FileNotFoundError(f"Config file does not exist: {config_path}")

    output_root = (
      args.output_root.resolve()
      if args.output_root is not None
      else (sim_dir / "processed" / "density")
    )

    configured_sort_names, nx, ny, nz, lx, ly, lz, diagnose_period, time_width = load_config(config_path)

    sort_names: list[str] = []
    sort_dirs: dict[str, Path] = {}
    missing_sort_names: list[str] = []
    for sort_name in configured_sort_names:
      density_dir = sim_dir / sort_name / "density"
      if not density_dir.exists():
        missing_sort_names.append(sort_name)
        continue
      sort_names.append(sort_name)
      sort_dirs[sort_name] = density_dir

    if missing_sort_names:
      print(
        "Warning: skipping density diagnostics for sorts without density directory: "
        + ", ".join(missing_sort_names)
      )

    if not sort_names:
      print(f"No density diagnostics found in '{sim_dir}'. Skipping density rendering.")
      return 0

    size_bytes = nx * ny * nz * np.dtype(np.float32).itemsize
    names = get_common_snapshot_names(
      sort_dirs,
      name_width=time_width,
      size_bytes=size_bytes,
    )
    sort_limits = compute_sort_limits(sort_dirs, names, nx, ny, nz)

    output_root.mkdir(parents=True, exist_ok=True)
    clear_existing_pngs(output_root)
    print(
      f"Density: rendering {len(names)} frame(s) for sorts: "
      + ", ".join(sort_names)
    )

    for frame_idx, name in enumerate(names):
      render_frame(
        frame_idx=frame_idx,
        name=name,
        sort_names=sort_names,
        sort_dirs=sort_dirs,
        sort_limits=sort_limits,
        nx=nx,
        ny=ny,
        nz=nz,
        lx=lx,
        ly=ly,
        lz=lz,
        diagnose_period=diagnose_period,
        out_dir=output_root,
        dpi=args.dpi,
        show=args.show,
      )

    return 0
  except Exception as exc:
    print(f"Error: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
