import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='.', help='Directory with boris.txt and drift_kinetic.txt')
args = parser.parse_args()

boris_data = np.loadtxt(os.path.join(args.dir, 'boris.txt'), skiprows=1)
dk_data = np.loadtxt(os.path.join(args.dir, 'drift_kinetic.txt'), skiprows=1)

# Извлечение координат из boris.txt: x (столбец 1), y (2), z (3)
x_b, y_b, z_b = boris_data[:, 1], boris_data[:, 2], boris_data[:, 3]

# Извлечение координат из drift_kinetic.txt: x (столбец 1), y (2), z (3)
x_dk, y_dk, z_dk = dk_data[:, 1], dk_data[:, 2], dk_data[:, 3]

# Создание фигуры с тремя подграфиками для разных срезов
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

def mark_endpoints(ax, xs, ys, color):
    ax.plot(xs[0], ys[0], 'o', color='white', markeredgecolor=color, markeredgewidth=1.5, markersize=7, zorder=5)
    ax.plot(xs[-1], ys[-1], 'o', color=color, markersize=7, zorder=5)

# 1. Срез XY
axs[0].plot(x_b, y_b, label='Boris', color='green')
axs[0].plot(x_dk, y_dk, label='DK', color='saddlebrown')
mark_endpoints(axs[0], x_b, y_b, 'green')
mark_endpoints(axs[0], x_dk, y_dk, 'saddlebrown')
axs[0].set_xlabel(r'$x\ [c/\omega_{pe}]$')
axs[0].set_ylim([-0.1, 2.0])
axs[0].set_ylabel(r'$y\ [c/\omega_{pe}]$')
axs[0].axhline(1.905, color='black', linestyle='--', linewidth=1)
axs[0].axhline(0.0, color='black', linestyle='--', linewidth=1)
axs[0].set_box_aspect(1)
axs[0].legend(loc='upper right')
axs[0].grid(True)

# 2. Срез XZ
axs[1].plot(x_b, z_b, label='Boris', color='green')
axs[1].plot(x_dk, z_dk, label='DK', color='saddlebrown')
mark_endpoints(axs[1], x_b, z_b, 'green')
mark_endpoints(axs[1], x_dk, z_dk, 'saddlebrown')
axs[1].set_xlabel(r'$x\ [c/\omega_{pe}]$')
axs[1].set_ylabel(r'$z\ [c/\omega_{pe}]$')
axs[1].set_box_aspect(1)
axs[1].legend()
axs[1].grid(True)

# 3. Срез YZ
axs[2].plot(y_b, z_b, label='Boris', color='green')
axs[2].plot(y_dk, z_dk, label='DK', color='saddlebrown')
mark_endpoints(axs[2], y_b, z_b, 'green')
mark_endpoints(axs[2], y_dk, z_dk, 'saddlebrown')
axs[2].set_xlabel(r'$y\ [c/\omega_{pe}]$')
axs[2].set_ylabel(r'$z\ [c/\omega_{pe}]$')
axs[2].set_box_aspect(1)
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
