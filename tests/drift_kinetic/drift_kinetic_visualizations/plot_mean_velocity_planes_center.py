#!/usr/bin/env python3

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLANES = ("XY", "XZ", "YZ")

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Render center XY/XZ/YZ mean-velocity vector slices for particle diagnostics."
  )
  parser.add_argument(
    "--sim-dir",
    type=Path,
    required=True,
    help="Simulation output directory with config.json and <sort>/{density,current}/",
  )
  parser.add_argument(
    "--output-root",
    type=Path,
    default=None,
    help="Output directory for PNG frames (default: <sim-dir>/processed/velocity)",
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
) -> tuple[list[tuple[str, float]], int, int, int, float, float, float, float, float, float, int]:
  with config_path.open("r", encoding="utf-8") as file:
    config = json.load(file)

  particles = config.get("Particles", [])
  sort_specs: list[tuple[str, float]] = []
  for item in particles:
    if "sort_name" not in item:
      continue
    sort_name = str(item["sort_name"])
    charge = parse_scalar(item["q"], {})
    sort_specs.append((sort_name, charge))

  if not sort_specs:
    raise ValueError("No particles with sort_name/q found in config.json")

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

  return sort_specs, nx, ny, nz, lx, ly, lz, dx, dy, dz, diagnose_period, time_width


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


def read_density_frame(path: Path, nx: int, ny: int, nz: int) -> np.ndarray:
  data = np.fromfile(path, dtype=np.float32)
  expected = nx * ny * nz
  if data.size != expected:
    raise ValueError(
      f"Unexpected density size in '{path}': {data.size}, expected {expected} "
      f"(nx={nx}, ny={ny}, nz={nz})"
    )
  return data.reshape((nz, ny, nx))


def read_current_frame(path: Path, nx: int, ny: int, nz: int) -> np.ndarray:
  data = np.fromfile(path, dtype=np.float32)
  expected = nx * ny * nz * 3
  if data.size != expected:
    raise ValueError(
      f"Unexpected current size in '{path}': {data.size}, expected {expected} "
      f"(nx={nx}, ny={ny}, nz={nz}, dof=3)"
    )
  return data.reshape((nz, ny, nx, 3))


def clear_existing_pngs(path: Path) -> None:
  if not path.exists():
    return
  for png in path.glob("*.png"):
    png.unlink()


def get_common_snapshot_names(
  sort_specs: list[tuple[str, float]],
  density_dirs: dict[str, Path],
  current_dirs: dict[str, Path],
  *,
  name_width: int,
  density_size_bytes: int,
  current_size_bytes: int,
) -> list[str]:
  common_names: set[str] | None = None

  for sort_name, _charge in sort_specs:
    density_names = {
      path.name for path in list_snapshot_files(
        density_dirs[sort_name],
        name_width=name_width,
        size_bytes=density_size_bytes,
      )
    }
    current_names = {
      path.name for path in list_snapshot_files(
        current_dirs[sort_name],
        name_width=name_width,
        size_bytes=current_size_bytes,
      )
    }
    sort_common = density_names.intersection(current_names)
    common_names = sort_common if common_names is None else common_names.intersection(sort_common)

  if not common_names:
    raise FileNotFoundError(
      "No common snapshot files among all particle sorts for density/current diagnostics."
    )

  return sorted(common_names, key=lambda name: (int(name), name))


def compute_velocity_frame(
  density_path: Path,
  current_path: Path,
  charge: float,
  nx: int,
  ny: int,
  nz: int,
) -> np.ndarray:
  if charge == 0.0:
    raise ValueError(f"Charge is zero for '{density_path.parent.parent.name}', cannot compute V = J / (q*n).")

  density = read_density_frame(density_path, nx, ny, nz)
  current = read_current_frame(current_path, nx, ny, nz)

  velocity = np.zeros_like(current, dtype=np.float32)
  density_scale = float(np.max(np.abs(density)))
  density_floor = max(density_scale * 1.0e-12, 1.0e-20)
  mask = np.abs(density) > density_floor

  velocity[mask] = current[mask] / (charge * density[mask, None])
  return velocity


def compute_sort_max_speed(
  sort_specs: list[tuple[str, float]],
  density_dirs: dict[str, Path],
  current_dirs: dict[str, Path],
  names: list[str],
  nx: int,
  ny: int,
  nz: int,
) -> dict[str, float]:
  limits: dict[str, float] = {}

  for sort_name, charge in sort_specs:
    max_speed = 0.0
    for name in names:
      velocity = compute_velocity_frame(
        density_dirs[sort_name] / name,
        current_dirs[sort_name] / name,
        charge,
        nx,
        ny,
        nz,
      )
      speed = np.linalg.norm(velocity, axis=-1)
      max_speed = max(max_speed, float(np.max(speed)))

    limits[sort_name] = max_speed

  return limits


def compute_stride(nx: int, ny: int, nz: int) -> int:
  return max(1, int(math.ceil(max(nx, ny, nz) / 24.0)))


def make_plane_vectors(
  velocity: np.ndarray,
  plane: str,
  x_coords: np.ndarray,
  y_coords: np.ndarray,
  z_coords: np.ndarray,
  x_center: int,
  y_center: int,
  z_center: int,
  stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, str]]:
  if plane == "XY":
    grid_x, grid_y = np.meshgrid(x_coords[::stride], y_coords[::stride], indexing="xy")
    u = velocity[z_center, ::stride, ::stride, 0]
    v = velocity[z_center, ::stride, ::stride, 1]
    return grid_x, grid_y, u, v, ("x [c/w_pe]", "y [c/w_pe]")

  if plane == "XZ":
    grid_x, grid_z = np.meshgrid(x_coords[::stride], z_coords[::stride], indexing="xy")
    u = velocity[::stride, y_center, ::stride, 0]
    v = velocity[::stride, y_center, ::stride, 2]
    return grid_x, grid_z, u, v, ("x [c/w_pe]", "z [c/w_pe]")

  if plane == "YZ":
    grid_y, grid_z = np.meshgrid(y_coords[::stride], z_coords[::stride], indexing="xy")
    u = velocity[::stride, ::stride, x_center, 1]
    v = velocity[::stride, ::stride, x_center, 2]
    return grid_y, grid_z, u, v, ("y [c/w_pe]", "z [c/w_pe]")

  raise ValueError(f"Unsupported plane: {plane}")


def safe_speed_limit(max_speed: float) -> float:
  return max_speed if max_speed > 0.0 else 1.0


def plane_target_length(plane: str, dx: float, dy: float, dz: float) -> float:
  if plane == "XY":
    return 0.75 * min(dx, dy)
  if plane == "XZ":
    return 0.75 * min(dx, dz)
  if plane == "YZ":
    return 0.75 * min(dy, dz)
  raise ValueError(f"Unsupported plane: {plane}")


def quiver_scale(max_speed: float, plane: str, dx: float, dy: float, dz: float) -> float:
  return safe_speed_limit(max_speed) / max(plane_target_length(plane, dx, dy, dz), 1.0e-12)


def render_frame(
  frame_idx: int,
  name: str,
  sort_specs: list[tuple[str, float]],
  density_dirs: dict[str, Path],
  current_dirs: dict[str, Path],
  sort_max_speed: dict[str, float],
  nx: int,
  ny: int,
  nz: int,
  lx: float,
  ly: float,
  lz: float,
  dx: float,
  dy: float,
  dz: float,
  diagnose_period: float,
  out_dir: Path,
  dpi: int,
  show: bool,
) -> None:
  nsorts = len(sort_specs)
  fig, axes = plt.subplots(
    nsorts,
    3,
    figsize=(15, max(4.2, 3.8 * nsorts)),
    constrained_layout=True,
  )

  if nsorts == 1:
    axes = np.asarray([axes])

  x_coords = (np.arange(nx, dtype=float) + 0.5) * dx
  y_coords = (np.arange(ny, dtype=float) + 0.5) * dy
  z_coords = (np.arange(nz, dtype=float) + 0.5) * dz
  x_center = nx // 2
  y_center = ny // 2
  z_center = nz // 2
  stride = compute_stride(nx, ny, nz)

  extents = {
    "XY": (0.0, lx, 0.0, ly),
    "XZ": (0.0, lx, 0.0, lz),
    "YZ": (0.0, ly, 0.0, lz),
  }

  for row, (sort_name, charge) in enumerate(sort_specs):
    velocity = compute_velocity_frame(
      density_dirs[sort_name] / name,
      current_dirs[sort_name] / name,
      charge,
      nx,
      ny,
      nz,
    )

    row_quiver = None
    for col, plane in enumerate(PLANES):
      axis = axes[row, col]
      grid_1, grid_2, u, v, labels = make_plane_vectors(
        velocity,
        plane,
        x_coords,
        y_coords,
        z_coords,
        x_center,
        y_center,
        z_center,
        stride,
      )

      speed = np.sqrt(u * u + v * v)
      speed_limit = safe_speed_limit(sort_max_speed[sort_name])
      row_quiver = axis.quiver(
        grid_1,
        grid_2,
        u,
        v,
        speed,
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=quiver_scale(speed_limit, plane, dx, dy, dz),
        pivot="mid",
        width=0.010,
        headwidth=5.5,
        headlength=7.0,
        headaxislength=6.0,
        minshaft=2.0,
        minlength=0.0,
      )
      row_quiver.set_clim(0.0, speed_limit)

      axis.set_xlim(extents[plane][0], extents[plane][1])
      axis.set_ylim(extents[plane][2], extents[plane][3])
      axis.set_aspect("auto")
      axis.set_box_aspect(1)
      axis.set_title(f"{sort_name}: {plane}")
      axis.set_xlabel(labels[0])
      axis.set_ylabel(labels[1])
      axis.grid(False)

    colorbar = fig.colorbar(
      row_quiver,
      ax=axes[row, :],
      shrink=0.95,
      fraction=0.03,
      pad=0.02,
    )
    colorbar.set_label(f"{sort_name} |V|")

    ref_speed = sort_max_speed[sort_name]
    if ref_speed > 0.0:
      axes[row, 2].quiverkey(
        row_quiver,
        X=0.98,
        Y=1.10,
        U=ref_speed,
        label=f"|V| = {ref_speed:.3g}",
        labelpos="E",
        coordinates="axes",
      )
    else:
      axes[row, 2].text(
        0.98,
        1.08,
        "max |V| = 0",
        ha="right",
        va="bottom",
        transform=axes[row, 2].transAxes,
      )

  time_value = frame_idx * diagnose_period
  fig.suptitle(
    (
      f"Mean velocity center slices, frame={frame_idx}, t={time_value:.6g} [1/w_pe], file={name}\n"
      f"V = J / (q*n), centers: z={z_center}, y={y_center}, x={x_center}, stride={stride}"
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
      else (sim_dir / "processed" / "velocity")
    )

    configured_sort_specs, nx, ny, nz, lx, ly, lz, dx, dy, dz, diagnose_period, time_width = load_config(
      config_path
    )

    sort_specs: list[tuple[str, float]] = []
    density_dirs: dict[str, Path] = {}
    current_dirs: dict[str, Path] = {}
    skipped_sort_messages: list[str] = []
    for sort_name, charge in configured_sort_specs:
      density_dir = sim_dir / sort_name / "density"
      current_dir = sim_dir / sort_name / "current"
      missing_parts: list[str] = []
      if not density_dir.exists():
        missing_parts.append("density")
      if not current_dir.exists():
        missing_parts.append("current")
      if missing_parts:
        skipped_sort_messages.append(f"{sort_name} ({', '.join(missing_parts)})")
        continue

      sort_specs.append((sort_name, charge))
      density_dirs[sort_name] = density_dir
      current_dirs[sort_name] = current_dir

    if skipped_sort_messages:
      print(
        "Warning: skipping mean-velocity diagnostics for sorts without complete density/current directories: "
        + ", ".join(skipped_sort_messages)
      )

    if not sort_specs:
      print(f"No complete density/current diagnostics found in '{sim_dir}'. Skipping mean-velocity rendering.")
      return 0

    density_size_bytes = nx * ny * nz * np.dtype(np.float32).itemsize
    current_size_bytes = nx * ny * nz * 3 * np.dtype(np.float32).itemsize
    names = get_common_snapshot_names(
      sort_specs,
      density_dirs,
      current_dirs,
      name_width=time_width,
      density_size_bytes=density_size_bytes,
      current_size_bytes=current_size_bytes,
    )
    sort_max_speed = compute_sort_max_speed(
      sort_specs,
      density_dirs,
      current_dirs,
      names,
      nx,
      ny,
      nz,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    clear_existing_pngs(output_root)
    print(
      f"Mean velocity: rendering {len(names)} frame(s) for sorts: "
      + ", ".join(sort_name for sort_name, _charge in sort_specs)
    )

    for frame_idx, name in enumerate(names):
      render_frame(
        frame_idx=frame_idx,
        name=name,
        sort_specs=sort_specs,
        density_dirs=density_dirs,
        current_dirs=current_dirs,
        sort_max_speed=sort_max_speed,
        nx=nx,
        ny=ny,
        nz=nz,
        lx=lx,
        ly=ly,
        lz=lz,
        dx=dx,
        dy=dy,
        dz=dz,
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
