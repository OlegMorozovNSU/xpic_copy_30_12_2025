#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLANE_AXIS_MAP = {
  "X": ("y", "z"),
  "Y": ("x", "z"),
  "Z": ("x", "y"),
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Plot a centered 1D |B| slice from a 2D FieldView snapshot."
  )
  parser.add_argument(
    "--input",
    type=Path,
    required=True,
    help="Path to one 2D field snapshot file such as .../B_planeX_0005/000000",
  )
  parser.add_argument(
    "--axis",
    choices=("x", "y", "z"),
    required=True,
    help="Global in-plane axis for the 1D center slice",
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


def load_geometry(config_path: Path) -> tuple[int, int, int, float, float, float]:
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
    "c/wpe": 1.0,
  }

  lx = parse_scalar(geometry["x"], scales)
  ly = parse_scalar(geometry["y"], scales)
  lz = parse_scalar(geometry["z"], scales)

  nx = int(round(lx / dx))
  ny = int(round(ly / dy))
  nz = int(round(lz / dz))
  return nx, ny, nz, dx, dy, dz


def parse_snapshot_path(input_path: Path) -> tuple[str, str, int]:
  match = re.fullmatch(r"(E|B)_plane([XYZ])_(\d+)", input_path.parent.name)
  if not match:
    raise ValueError(
      "Input parent directory must match '<field>_plane<axis>_<index>', "
      f"got '{input_path.parent.name}'."
    )

  field = match.group(1)
  plane = match.group(2)
  plane_index = int(match.group(3))
  return field, plane, plane_index


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
      f"{expected * np.dtype(np.float32).itemsize} or {expected * np.dtype(np.float64).itemsize}."
    )

  data = np.fromfile(path, dtype=dtype)
  if data.size != expected:
    raise ValueError(f"Unexpected size in '{path}': {data.size}, expected {expected} for plane {plane}.")
  return data.reshape(shape)


def extract_centerline(
  magnitude: np.ndarray,
  plane: str,
  axis_name: str,
  nx: int,
  ny: int,
  nz: int,
  dx: float,
  dy: float,
  dz: float,
) -> tuple[np.ndarray, np.ndarray, str]:
  valid_axes = PLANE_AXIS_MAP[plane]
  if axis_name not in valid_axes:
    valid_axes_text = ", ".join(valid_axes)
    raise ValueError(
      f"Axis '{axis_name}' is not in plane{plane}. Valid axes for this input: {valid_axes_text}."
    )

  if plane == "X":
    if axis_name == "y":
      center_idx = nz // 2
      coordinates = np.arange(ny, dtype=float) * dy
      values = magnitude[center_idx, :]
      center_text = f"z={center_idx * dz:.6g} [c/w_pe]"
    else:
      center_idx = ny // 2
      coordinates = np.arange(nz, dtype=float) * dz
      values = magnitude[:, center_idx]
      center_text = f"y={center_idx * dy:.6g} [c/w_pe]"
  elif plane == "Y":
    if axis_name == "x":
      center_idx = nz // 2
      coordinates = np.arange(nx, dtype=float) * dx
      values = magnitude[center_idx, :]
      center_text = f"z={center_idx * dz:.6g} [c/w_pe]"
    else:
      center_idx = nx // 2
      coordinates = np.arange(nz, dtype=float) * dz
      values = magnitude[:, center_idx]
      center_text = f"x={center_idx * dx:.6g} [c/w_pe]"
  else:
    if axis_name == "x":
      center_idx = ny // 2
      coordinates = np.arange(nx, dtype=float) * dx
      values = magnitude[center_idx, :]
      center_text = f"y={center_idx * dy:.6g} [c/w_pe]"
    else:
      center_idx = nx // 2
      coordinates = np.arange(ny, dtype=float) * dy
      values = magnitude[:, center_idx]
      center_text = f"x={center_idx * dx:.6g} [c/w_pe]"

  return coordinates, values, center_text


def fixed_plane_position(plane: str, plane_index: int, dx: float, dy: float, dz: float) -> str:
  if plane == "X":
    return f"x={plane_index * dx:.6g} [c/w_pe]"
  if plane == "Y":
    return f"y={plane_index * dy:.6g} [c/w_pe]"
  if plane == "Z":
    return f"z={plane_index * dz:.6g} [c/w_pe]"
  raise ValueError(f"Unsupported plane: {plane}")


def main() -> int:
  args = parse_args()

  input_path = args.input.resolve()
  if not input_path.exists():
    raise FileNotFoundError(f"Input snapshot does not exist: {input_path}")
  if not input_path.is_file():
    raise ValueError(f"Input path is not a file: {input_path}")

  config_path = input_path.parent.parent / "config.json"
  if not config_path.exists():
    raise FileNotFoundError(f"Config file does not exist: {config_path}")

  field, plane, plane_index = parse_snapshot_path(input_path)
  nx, ny, nz, dx, dy, dz = load_geometry(config_path)
  frame = read_plane_frame(input_path, plane, nx, ny, nz)
  magnitude = np.linalg.norm(frame, axis=2)

  coordinates, values, center_text = extract_centerline(
    magnitude,
    plane,
    args.axis,
    nx,
    ny,
    nz,
    dx,
    dy,
    dz,
  )
  plane_text = fixed_plane_position(plane, plane_index, dx, dy, dz)

  fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
  axis.plot(coordinates, values, linewidth=2.0)
  axis.set_xlabel(f"{args.axis} [c/w_pe]")
  axis.set_ylabel(f"|{field}|")
  axis.grid(True, alpha=0.3)
  axis.set_title(
    f"{field} centerline from plane{plane}, file={input_path.name}\n"
    f"{plane_text}, centered at {center_text}"
  )

  plt.show()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
