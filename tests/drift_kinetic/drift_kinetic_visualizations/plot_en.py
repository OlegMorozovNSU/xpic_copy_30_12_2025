#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute energy and plot relative change ."
    )
    parser.add_argument("--input", required=True, help="Path to input file")
    parser.add_argument("--number", type=int, required=True,
                        help="Number of points to plot (starting from 2nd row)")
    return parser.parse_args()

def read_file(path):
    t_list = []
    dK_list = []
    W_list = []

    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    if len(lines) < 2:
        print("Error: file must contain header + data", file=sys.stderr)
        sys.exit(1)

    # пропускаем заголовок
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) < 20:
            continue

        try:
            t = float(parts[0])
            p_par = float(parts[9])
            p_perp = float(parts[19])
        except ValueError:
            continue

        t_list.append(t)
        dK_list.append(p_par)
        W_list.append(p_perp)

    return np.array(t_list), np.array(dK_list), np.array(W_list)

def main():
    args = parse_args()

    if args.number <= 0:
        print("Error: --number must be > 0", file=sys.stderr)
        sys.exit(1)

    t, _dK, _W = read_file(args.input)

    if len(t) < 2:
        print("Error: not enough data points", file=sys.stderr)
        sys.exit(1)

    mean = np.mean(_dK)

    dK =_dK/_W[0]/10.

    dK = np.cumulative_sum(dK)

    rel = dK

    # пропускаем первый шаг

    plt.figure(figsize=(8, 5))
    plt.plot(t, rel, marker="o", color='green')
    plt.xlabel(r"t, $\tau\omega_{pe}$", fontsize=13)
    plt.ylabel(r"$\frac{\Delta W}{\tau}$", fontsize=13)
    #plt.xlim([0,2000])
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    rel = _W

    plt.figure(figsize=(8, 5))
    plt.plot(t[1:-1]-1, rel[1:-1], marker="o", color='saddlebrown')
    plt.xlabel(r"t, $\tau\omega_{pe}$", fontsize=13)
    plt.ylabel(r'$\left\| \partial_t \rho + \nabla \cdot \mathbf{J} \right\|$', fontsize=13)
    plt.xlim([0,2000])
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
