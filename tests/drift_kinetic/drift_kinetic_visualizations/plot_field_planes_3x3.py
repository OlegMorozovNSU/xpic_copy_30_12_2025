#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


PLANES = ("X", "Y", "Z")
COMPONENTS = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
  script_dir = Path(__file__).resolve().parent
  default_sim_dir = (script_dir / ".." / "output" / "drift_kinetic_alfen_ex1").resolve()

  parser = argparse.ArgumentParser(
    description="Render 3x3 field-view snapshots for available E and B diagnostics."
  )
  parser.add_argument(
    "--sim-dir",
    type=Path,
    default=default_sim_dir,
    help="Simulation output directory with config.json and FieldView folders",
  )
  parser.add_argument(
    "--output-root",
    type=Path,
    default=None,
    help="Output root for PNG frames (default: <sim-dir>/processed/fields)",
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


def load_geometry(
  config_path: Path,
) -> tuple[int, int, int, float, float, float, float, float, float, float, float, int]:
  with config_path.open("r", encoding="utf-8") as file:
    config = json.load(file)

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
  return nx, ny, nz, lx, ly, lz, dx, dy, dz, dt, diagnose_period, time_width


def load_background_field(config_path: Path, target_field: str) -> Optional[np.ndarray]:
  with config_path.open("r", encoding="utf-8") as file:
    config = json.load(file)

  background = np.zeros(3, dtype=float)
  found = False
  for section_key in ("Presets", "Commands", "StepPresets"):
    for cmd in config.get(section_key, []) or []:
      if cmd.get("command") != "SetMagneticField":
        continue
      if cmd.get("field_axpy") != target_field and cmd.get("field") != target_field:
        continue
      setter = cmd.get("setter", {})
      if setter.get("name") != "SetUniformField":
        print(
          f"Warning: non-uniform background setter '{setter.get('name')}' for "
          f"{target_field} is not supported; skipping."
        )
        continue
      value = setter.get("value", [0.0, 0.0, 0.0])
      background = background + np.asarray(value, dtype=float)
      found = True

  return background if found else None


def list_snapshot_files(
  path: Path,
  *,
  name_width: int,
  size_bytes: int | tuple[int, ...],
) -> list[Path]:
  allowed_sizes = (size_bytes,) if isinstance(size_bytes, int) else size_bytes
  allowed_size_set = set(allowed_sizes)
  files = [
    item for item in path.iterdir()
    if item.is_file()
    and item.name.isdigit()
    and item.stat().st_size in allowed_size_set
  ]
  if not files:
    allowed_sizes_text = ", ".join(str(size) for size in sorted(allowed_size_set))
    raise FileNotFoundError(
      f"No valid snapshot files found in '{path}' with allowed byte sizes: {allowed_sizes_text}"
    )

  def key(file_path: Path) -> tuple[int, str]:
    name = file_path.name
    return (int(name), name)

  return sorted(files, key=key)


def pick_plane_directory(sim_dir: Path, field: str, plane: str) -> Optional[tuple[Path, int]]:
  pattern = re.compile(rf"^{re.escape(field)}_plane{plane}_(\d+)$")
  candidates: list[tuple[int, Path]] = []

  for item in sim_dir.iterdir():
    if not item.is_dir():
      continue
    match = pattern.match(item.name)
    if match:
      candidates.append((int(match.group(1)), item))

  if not candidates:
    return None

  candidates.sort(key=lambda pair: pair[0])
  if len(candidates) > 1:
    print(
      f"Warning: multiple directories found for {field} plane {plane}; "
      f"using '{candidates[-1][1].name}'."
    )

  return candidates[-1][1], candidates[-1][0]


def collect_plane_directories(
  sim_dir: Path,
  field: str,
) -> tuple[dict[str, Path], dict[str, int], list[str]]:
  plane_dirs: dict[str, Path] = {}
  plane_indices: dict[str, int] = {}
  missing_planes: list[str] = []

  for plane in PLANES:
    result = pick_plane_directory(sim_dir, field, plane)
    if result is None:
      missing_planes.append(plane)
      continue
    directory, index = result
    plane_dirs[plane] = directory
    plane_indices[plane] = index

  return plane_dirs, plane_indices, missing_planes


def ordered_available_planes(plane_dirs: dict[str, Path]) -> list[str]:
  return [plane for plane in PLANES if plane in plane_dirs]


def get_common_snapshot_names(
  plane_dirs: dict[str, Path],
  *,
  nx: int,
  ny: int,
  nz: int,
  time_width: int,
) -> list[str]:
  expected_sizes = {
    "X": (
      ny * nz * 3 * np.dtype(np.float32).itemsize,
      ny * nz * 3 * np.dtype(np.float64).itemsize,
    ),
    "Y": (
      nx * nz * 3 * np.dtype(np.float32).itemsize,
      nx * nz * 3 * np.dtype(np.float64).itemsize,
    ),
    "Z": (
      nx * ny * 3 * np.dtype(np.float32).itemsize,
      nx * ny * 3 * np.dtype(np.float64).itemsize,
    ),
  }
  available_planes = ordered_available_planes(plane_dirs)
  if not available_planes:
    raise FileNotFoundError("No plane directories were found for field diagnostics.")

  common_names: Optional[set[str]] = None
  for plane in available_planes:
    names = {
      path.name for path in list_snapshot_files(
        plane_dirs[plane],
        name_width=time_width,
        size_bytes=expected_sizes[plane],
      )
    }
    common_names = names if common_names is None else common_names.intersection(names)

  if not common_names:
    raise FileNotFoundError(
      "No common snapshots between available planes for one of fields."
    )

  return sorted(common_names, key=lambda name: (int(name), name))


def read_plane_frame(path: Path, plane: str, nx: int, ny: int, nz: int) -> np.ndarray:
  if plane == "X":
    expected = ny * nz * 3
    shape = (nz, ny, 3)
  elif plane == "Y":
    expected = nx * nz * 3
    shape = (nz, nx, 3)
  elif plane == "Z":
    expected = nx * ny * 3
    shape = (ny, nx, 3)
  else:
    raise ValueError(f"Unsupported plane: {plane}")

  file_size = path.stat().st_size
  if file_size == expected * np.dtype(np.float32).itemsize:
    dtype = np.float32
  elif file_size == expected * np.dtype(np.float64).itemsize:
    dtype = np.float64
  else:
    raise ValueError(
      f"Unexpected byte size in '{path}': {file_size}, expected one of "
      f"{expected * np.dtype(np.float32).itemsize} or {expected * np.dtype(np.float64).itemsize}"
    )

  data = np.fromfile(path, dtype=dtype)
  if data.size != expected:
    raise ValueError(f"Unexpected size in '{path}': {data.size}, expected {expected} for plane {plane}.")
  return data.reshape(shape)


def compute_component_limits(
  plane_dirs: dict[str, Path],
  names: list[str],
  nx: int,
  ny: int,
  nz: int,
  background: Optional[np.ndarray] = None,
) -> np.ndarray:
  limits = np.zeros(3, dtype=float)
  available_planes = ordered_available_planes(plane_dirs)
  for name in names:
    for plane in available_planes:
      frame = read_plane_frame(plane_dirs[plane] / name, plane, nx, ny, nz)
      data = frame if background is None else frame - background
      for comp in range(3):
        limits[comp] = max(limits[comp], float(np.max(np.abs(data[:, :, comp]))))

  limits[limits == 0.0] = 1.0
  return limits


def compute_ratio_limit(
  plane_dirs: dict[str, Path],
  names: list[str],
  nx: int,
  ny: int,
  nz: int,
  background: np.ndarray,
) -> float:
  b0_mag = float(np.linalg.norm(background))
  if b0_mag == 0.0:
    return 1.0
  b0_hat = background / b0_mag

  limit = 0.0
  available_planes = ordered_available_planes(plane_dirs)
  for name in names:
    for plane in available_planes:
      frame = read_plane_frame(plane_dirs[plane] / name, plane, nx, ny, nz)
      delta = frame - background
      ratio = np.tensordot(delta, b0_hat, axes=([-1], [0])) / b0_mag
      limit = max(limit, float(np.max(np.abs(ratio))))

  return limit if limit > 0.0 else 1.0


def clear_existing_pngs(path: Path) -> None:
  if not path.exists():
    return
  for png in path.glob("*.png"):
    png.unlink()


def render_field_frame(
  field: str,
  frame_idx: int,
  name: str,
  plane_dirs: dict[str, Path],
  plane_indices: dict[str, int],
  out_dir: Path,
  component_limits: np.ndarray,
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
  dpi: int,
  show: bool,
  background: Optional[np.ndarray] = None,
  ratio_limit: Optional[float] = None,
) -> None:
  available_planes = ordered_available_planes(plane_dirs)
  frames = {
    plane: read_plane_frame(plane_dirs[plane] / name, plane, nx, ny, nz)
    for plane in available_planes
  }

  layout: dict[str, dict[str, object]] = {}
  if "X" in plane_indices:
    layout["X"] = {
      "slice": "yz",
      "coord_value": plane_indices["X"] * dx,
      "extent": (0.0, ly, 0.0, lz),
      "xlabel": "y [c/w_pe]",
      "ylabel": "z [c/w_pe]",
    }
  if "Y" in plane_indices:
    layout["Y"] = {
      "slice": "xz",
      "coord_value": plane_indices["Y"] * dy,
      "extent": (0.0, lx, 0.0, lz),
      "xlabel": "x [c/w_pe]",
      "ylabel": "z [c/w_pe]",
    }
  if "Z" in plane_indices:
    layout["Z"] = {
      "slice": "xy",
      "coord_value": plane_indices["Z"] * dz,
      "extent": (0.0, lx, 0.0, ly),
      "xlabel": "x [c/w_pe]",
      "ylabel": "y [c/w_pe]",
    }

  show_ratio = background is not None
  ncols = 4 if show_ratio else 3
  fig, axes = plt.subplots(
    len(available_planes),
    ncols,
    figsize=(3.5 * ncols + 2.0, max(4.2, 3.8 * len(available_planes))),
    constrained_layout=True,
  )
  if len(available_planes) == 1:
    axes = np.asarray([axes])

  column_mappables: dict[int, plt.Axes] = {}

  comp_prefix = "\u03b4" if background is not None else ""

  for row, plane in enumerate(available_planes):
    plane_info = layout[plane]
    frame = frames[plane]
    comp_data = frame if background is None else frame - background
    for col, comp_name in enumerate(COMPONENTS):
      im = axes[row, col].imshow(
        comp_data[:, :, col],
        origin="lower",
        aspect="auto",
        extent=plane_info["extent"],
        cmap="RdBu_r",
        vmin=-component_limits[col],
        vmax=component_limits[col],
      )
      if row == 0:
        column_mappables[col] = im
      axes[row, col].set_title(
        f"{plane_info['slice']} plane, {comp_prefix}{field}{comp_name}"
      )
      axes[row, col].set_xlabel(plane_info["xlabel"])
      axes[row, col].set_ylabel(plane_info["ylabel"])
      axes[row, col].grid(False)

    if show_ratio:
      b0_mag = float(np.linalg.norm(background))
      if b0_mag > 0.0:
        b0_hat = background / b0_mag
        delta = frame - background
        ratio = np.tensordot(delta, b0_hat, axes=([-1], [0])) / b0_mag
      else:
        ratio = np.zeros(frame.shape[:-1], dtype=float)
      vmax = ratio_limit if (ratio_limit is not None and ratio_limit > 0.0) else None
      im = axes[row, 3].imshow(
        ratio,
        origin="lower",
        aspect="auto",
        extent=plane_info["extent"],
        cmap="RdBu_r",
        vmin=-vmax if vmax is not None else None,
        vmax=vmax,
      )
      if row == 0:
        column_mappables[3] = im
      axes[row, 3].set_title(
        f"{plane_info['slice']} plane, \u03b4{field}\u2225/|{field}0|"
      )
      axes[row, 3].set_xlabel(plane_info["xlabel"])
      axes[row, 3].set_ylabel(plane_info["ylabel"])
      axes[row, 3].grid(False)

  for col, comp_name in enumerate(COMPONENTS):
    colorbar = fig.colorbar(
      column_mappables[col],
      ax=axes[:, col],
      shrink=0.92,
      fraction=0.03,
      pad=0.02,
    )
    colorbar.set_label(f"{comp_prefix}{field}{comp_name}")

  if show_ratio:
    colorbar = fig.colorbar(
      column_mappables[3],
      ax=axes[:, 3],
      shrink=0.92,
      fraction=0.03,
      pad=0.02,
    )
    colorbar.set_label(f"\u03b4{field}\u2225/|{field}0|")

  time_value = frame_idx * diagnose_period
  plane_positions = ", ".join(
    f"{plane}: {plane.lower()}={layout[plane]['coord_value']:.6g}"
    for plane in available_planes
  )
  fig.suptitle(
    (
      f"{field} field, frame={frame_idx}, t={time_value:.6g} [1/w_pe], file={name}\n"
      f"{plane_positions} [c/w_pe]"
    ),
    fontsize=12,
  )

  out_path = out_dir / f"{name}.png"
  fig.savefig(out_path, dpi=dpi)
  if show:
    plt.show()
  plt.close(fig)

  print(f"Saved: {out_path} (frame={frame_idx}, t={time_value:.6g})")


def render_field(
  sim_dir: Path,
  output_root: Path,
  field: str,
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
  time_width: int,
  dpi: int,
  show: bool,
  background: Optional[np.ndarray] = None,
) -> bool:
  plane_dirs, plane_indices, missing_planes = collect_plane_directories(sim_dir, field)
  available_planes = ordered_available_planes(plane_dirs)
  if not available_planes:
    missing = ", ".join(missing_planes)
    print(
      f"Warning: skipping {field} field diagnostics because plane directories "
      f"for {missing} are missing in '{sim_dir}'."
    )
    return False

  if missing_planes:
    missing = ", ".join(missing_planes)
    print(
      f"Warning: {field} field diagnostics missing planes {missing} in '{sim_dir}'. "
      f"Rendering available planes only."
    )

  names = get_common_snapshot_names(
    plane_dirs,
    nx=nx,
    ny=ny,
    nz=nz,
    time_width=time_width,
  )
  component_limits = compute_component_limits(plane_dirs, names, nx, ny, nz, background)

  ratio_limit: Optional[float] = None
  if background is not None:
    ratio_limit = compute_ratio_limit(plane_dirs, names, nx, ny, nz, background)

  out_dir = output_root / field
  out_dir.mkdir(parents=True, exist_ok=True)
  clear_existing_pngs(out_dir)

  print(
    f"{field}: rendering {len(names)} frame(s), plane directories: "
    + ", ".join(f"{plane}={plane_dirs[plane].name}" for plane in available_planes)
  )
  if background is not None:
    print(
      f"{field}: subtracting background {tuple(background.tolist())} for \u03b4{field}\u2225/|{field}0| panel"
    )
  for frame_idx, name in enumerate(names):
    render_field_frame(
      field=field,
      frame_idx=frame_idx,
      name=name,
      plane_dirs=plane_dirs,
      plane_indices=plane_indices,
      out_dir=out_dir,
      component_limits=component_limits,
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
      dpi=dpi,
      show=show,
      background=background,
      ratio_limit=ratio_limit,
    )

  return True


def main() -> int:
  args = parse_args()

  sim_dir = args.sim_dir.resolve()
  config_path = sim_dir / "config.json"
  if not config_path.exists():
    raise FileNotFoundError(f"Config file does not exist: {config_path}")

  output_root = (
    args.output_root.resolve()
    if args.output_root is not None
    else (sim_dir / "processed" / "fields")
  )

  nx, ny, nz, lx, ly, lz, dx, dy, dz, _dt, diagnose_period, time_width = load_geometry(config_path)
  output_root.mkdir(parents=True, exist_ok=True)

  rendered_fields: list[str] = []
  for field in ("E", "B"):
    background = load_background_field(config_path, field) if field == "B" else None
    if render_field(
      sim_dir=sim_dir,
      output_root=output_root,
      field=field,
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
      time_width=time_width,
      dpi=args.dpi,
      show=args.show,
      background=background,
    ):
      rendered_fields.append(field)

  if not rendered_fields:
    print(f"No complete E/B field diagnostics found in '{sim_dir}'. Skipping field rendering.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
