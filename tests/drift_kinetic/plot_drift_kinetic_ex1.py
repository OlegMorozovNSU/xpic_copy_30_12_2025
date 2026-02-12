#!/usr/bin/env python3

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLUMNS = ("Time", "dE+dB+dK", "N1dQ_tot")


def parse_args() -> argparse.Namespace:
  script_dir = os.path.dirname(os.path.abspath(__file__))

  default_input = os.path.join(
    script_dir,
    "output",
    "drift_kinetic_ex1",
    "temporal",
    "dk_diagnostic.txt",
  )
  default_output = os.path.join(
    script_dir,
    "output",
    "drift_kinetic_ex1",
    "temporal",
    "dk_diagnostic_conservation.png",
  )

  parser = argparse.ArgumentParser(
    description=(
      "Визуализация невязок законов сохранения для drift_kinetic_ex1"
    )
  )
  parser.add_argument(
    "--input",
    default=default_input,
    help="Path to input dk_diagnostic.txt",
  )
  parser.add_argument(
    "--output",
    default=default_output,
    help="Path to output PNG",
  )
  parser.add_argument(
    "--show",
    action="store_true",
    help="Show the plot window after saving",
  )
  parser.add_argument(
    "--dpi",
    type=int,
    default=150,
    help="Output image DPI",
  )
  parser.add_argument(
    "--config",
    default=None,
    help="Path to config.json (default: autodetect from --input)",
  )
  return parser.parse_args()


def read_diagnostic(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  if not os.path.exists(path):
    raise FileNotFoundError(f"Input file does not exist: {path}")

  with open(path, "r", encoding="utf-8") as file:
    header = file.readline().split()

  if not header:
    raise ValueError(f"Input file is empty or has no header: {path}")

  missing = [name for name in REQUIRED_COLUMNS if name not in header]
  if missing:
    raise ValueError(
      "Missing required columns in header: " + ", ".join(missing)
    )

  try:
    data = np.loadtxt(path, skiprows=1)
  except Exception as exc:
    raise ValueError(f"Failed to parse numeric data from '{path}': {exc}") from exc

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

  time = data[:, column_index["Time"]]
  energy_residual = data[:, column_index["dE+dB+dK"]]
  charge_residual = data[:, column_index["N1dQ_tot"]]

  return time, energy_residual, charge_residual


def get_config_path(input_path: str, config_path: str | None) -> str:
  if config_path:
    return config_path

  # <...>/drift_kinetic_ex1/temporal/dk_diagnostic.txt -> <...>/drift_kinetic_ex1/config.json
  sim_dir = os.path.dirname(os.path.dirname(os.path.abspath(input_path)))
  return os.path.join(sim_dir, "config.json")


def read_simulation_time_info(config_path: str) -> tuple[float, float]:
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file does not exist: {config_path}")

  try:
    with open(config_path, "r", encoding="utf-8") as file:
      config = json.load(file)
  except Exception as exc:
    raise ValueError(f"Failed to parse config JSON '{config_path}': {exc}") from exc

  try:
    geometry = config["Geometry"]
    dt = float(geometry["dt"])
    model_time = float(geometry["t"])
  except Exception as exc:
    raise ValueError(
      f"Failed to read Geometry.dt and Geometry.t from '{config_path}': {exc}"
    ) from exc

  if dt <= 0.0:
    raise ValueError(f"Geometry.dt must be positive, got: {dt}")
  if model_time <= 0.0:
    raise ValueError(f"Geometry.t must be positive, got: {model_time}")

  return dt, model_time


def build_plot(
  time: np.ndarray,
  energy_residual: np.ndarray,
  charge_residual: np.ndarray,
  dt: float,
  model_time: float,
) -> plt.Figure:
  fig, ax_energy = plt.subplots(figsize=(10, 5))
  ax_charge = ax_energy.twinx()

  line_energy = ax_energy.plot(
    time,
    energy_residual,
    color="tab:blue",
    linewidth=1.8,
    label=r"$\Delta W_E + \Delta W_B + \Delta W_K$",
  )
  line_charge = ax_charge.plot(
    time,
    charge_residual,
    color="tab:red",
    linewidth=1.8,
    label=r"$\Delta Q$",
  )

  ax_energy.set_xlim(0.0, model_time)

  ax_energy.set_xlabel(r"Время, $\omega_p t$")
  ax_energy.set_ylabel(r"Сохранение энергии: $\Delta W_E + \Delta W_B + \Delta W_K$", color="tab:blue")
  ax_charge.set_ylabel(r"Сохранение заряда: $\Delta Q$", color="tab:red")

  ax_energy.tick_params(axis="y", labelcolor="tab:blue")
  ax_charge.tick_params(axis="y", labelcolor="tab:red")

  ax_energy.grid(True, alpha=0.35)
  ax_energy.set_title(
    r"Ошибки законов сохранения, "
    + rf"$dt = {dt:g}\,[1/\omega_p]$"
  )

  lines = line_energy + line_charge
  labels = [line.get_label() for line in lines]
  ax_energy.legend(lines, labels, loc="best")

  fig.tight_layout()
  return fig


def main() -> int:
  args = parse_args()

  try:
    time_idx, energy_residual, charge_residual = read_diagnostic(args.input)

    config_path = get_config_path(args.input, args.config)
    dt, model_time = read_simulation_time_info(config_path)

    # Time column in dk_diagnostic is timestep index.
    time = time_idx * dt
    figure = build_plot(time, energy_residual, charge_residual, dt, model_time)

    output_dir = os.path.dirname(args.output)
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)

    figure.savefig(args.output, dpi=args.dpi)
    print(f"Saved: {args.output}")

    if args.show:
      plt.show()

    plt.close(figure)
    return 0

  except Exception as exc:
    print(f"Error: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
