#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLUMNS = ("Time", "dE", "dB", "wE", "wB")


def parse_args() -> argparse.Namespace:
  script_dir = Path(__file__).resolve().parent
  default_sim_dir = (script_dir / ".." / "output" / "drift_kinetic_ex3").resolve()

  parser = argparse.ArgumentParser(
    description=(
      "Plot cumulative sums of dE, dB and dK_<sort> normalized by initial energy values."
    )
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
    help="Path to output PNG (default: <sim-dir>/processed/dk/dk_diagnostic_component_sums.png)",
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


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
  sim_dir = args.sim_dir.resolve()

  input_path = args.input.resolve() if args.input is not None else (sim_dir / "temporal" / "dk_diagnostic.txt")
  if args.output is not None:
    output_path = args.output.resolve()
  else:
    inferred_sim_dir = input_path.parent.parent
    output_path = inferred_sim_dir / "processed" / "dk" / "dk_diagnostic_component_sums.png"

  return input_path, output_path


def read_diagnostic(
  path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, float]]:
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
  max_required_idx = max(column_index[name] for name in REQUIRED_COLUMNS)
  if data.shape[1] <= max_required_idx:
    raise ValueError(
      "Data columns count is smaller than header-required indexes: "
      f"shape={data.shape}, max_required_index={max_required_idx}"
    )

  time = data[:, column_index["Time"]]
  d_e = data[:, column_index["dE"]]
  d_b = data[:, column_index["dB"]]
  normalizers = {
    "dE": float(data[0, column_index["wE"]]),
    "dB": float(data[0, column_index["wB"]]),
  }

  d_k_by_sort: dict[str, np.ndarray] = {}
  for name in header:
    if name.startswith("dK_"):
      idx = column_index[name]
      if idx < data.shape[1]:
        d_k_by_sort[name] = data[:, idx]
        weight_name = "wK_" + name[3:]
        if weight_name not in column_index:
          raise ValueError(f"Missing required column for normalization: {weight_name}")
        normalizers[name] = float(data[0, column_index[weight_name]])

  if not d_k_by_sort:
    if "dK" not in column_index:
      raise ValueError("No dK_<sort> columns found and total dK column is missing.")
    d_k_by_sort["dK"] = data[:, column_index["dK"]]
    if "wK" not in column_index:
      raise ValueError("Missing required column for normalization: wK")
    normalizers["dK"] = float(data[0, column_index["wK"]])

  return time, d_e, d_b, d_k_by_sort, normalizers


def normalize_if_nonzero(values: np.ndarray, normalizer: float) -> np.ndarray:
  if normalizer == 0.0:
    return values
  return values / normalizer


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


def compute_symlog_linthresh(values: list[np.ndarray]) -> float:
  finite_chunks = [
    np.abs(np.asarray(chunk, dtype=float)[np.isfinite(chunk)])
    for chunk in values
    if chunk.size > 0
  ]
  if not finite_chunks:
    return 1.0

  abs_values = np.concatenate(finite_chunks)
  nonzero = abs_values[abs_values > 0.0]
  if nonzero.size == 0:
    return 1.0

  return float(10.0 ** np.floor(np.log10(np.min(nonzero))))


def sort_label_from_column(name: str) -> str:
  if name.startswith("dK_"):
    return name[3:]
  return name


def build_plot(
  time: np.ndarray,
  d_e_sum: np.ndarray,
  d_b_sum: np.ndarray,
  d_k_sums: dict[str, np.ndarray],
) -> plt.Figure:
  fig, (ax_linear, ax_log) = plt.subplots(
    2,
    1,
    figsize=(11, 9),
    sharex=True,
  )

  series: list[tuple[str, np.ndarray]] = [
    (r"$\sum dE / wE_0$", d_e_sum),
    (r"$\sum dB / wB_0$", d_b_sum),
  ]
  for key in sorted(d_k_sums):
    sort_name = sort_label_from_column(key)
    if key.startswith("dK_"):
      series.append((rf"$\sum dK$ ({sort_name}) / wK_0", d_k_sums[key]))
    else:
      series.append((r"$\sum dK / wK_0$", d_k_sums[key]))

  for idx, (label, values) in enumerate(series):
    color = f"C{idx % 10}"
    ax_linear.plot(
      time,
      values,
      linewidth=1.6,
      color=color,
      label=label,
    )
    ax_log.plot(
      time,
      values,
      linewidth=1.6,
      color=color,
      label=label,
    )

  set_x_limits_to_available_data(ax_linear, time)
  set_x_limits_to_available_data(ax_log, time)

  ax_linear.set_ylabel("Relative cumulative change")
  ax_linear.set_title("Relative cumulative dE, dB, dK by particle sort")
  ax_linear.axhline(0.0, color="black", linewidth=0.9, alpha=0.45)
  ax_linear.grid(True, which="both", alpha=0.35)
  ax_linear.legend(loc="best")

  linthresh = compute_symlog_linthresh([values for _label, values in series])
  ax_log.set_yscale("symlog", base=10, linthresh=linthresh)
  ax_log.set_ylabel("Relative cumulative change (symlog y)")
  ax_log.set_xlabel("Time")
  ax_log.axhline(0.0, color="black", linewidth=0.9, alpha=0.45)
  ax_log.grid(True, which="both", alpha=0.35)
  ax_log.legend(loc="best")

  fig.tight_layout()
  return fig


def main() -> int:
  args = parse_args()

  try:
    input_path, output_path = resolve_paths(args)
    time, d_e, d_b, d_k_by_sort, normalizers = read_diagnostic(input_path)

    d_e_sum = normalize_if_nonzero(np.cumsum(d_e), normalizers["dE"])
    d_b_sum = normalize_if_nonzero(np.cumsum(d_b), normalizers["dB"])
    d_k_sums = {
      name: normalize_if_nonzero(np.cumsum(values), normalizers[name])
      for name, values in d_k_by_sort.items()
    }

    figure = build_plot(
      time=time,
      d_e_sum=d_e_sum,
      d_b_sum=d_b_sum,
      d_k_sums=d_k_sums,
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
