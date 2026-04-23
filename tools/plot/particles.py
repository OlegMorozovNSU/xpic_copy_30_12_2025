#!/usr/bin/env python3

from plot import *

init_constants("results/ecsimcorr_dx8.0_dt2.0_Np1000/ecsimcorr_dx8.0_dt2.0_Np1000.json")
const.B0 = float(const.config["Presets"][0]["setter"]["value"][2])
const.tau = read_scalar(const.config["StepPresets"][3]["tau"])
const.min_x = read_scalar(const.config["StepPresets"][0]["geometry"]["min"][0]) - const.Lx / 2
const.max_x = read_scalar(const.config["StepPresets"][0]["geometry"]["max"][0]) - const.Lx / 2
const.min_y = read_scalar(const.config["StepPresets"][0]["geometry"]["min"][1]) 
const.max_y = read_scalar(const.config["StepPresets"][0]["geometry"]["max"][1])
const.pto = 10 * const.Ndts

def read(file_t, c, cd):
    data_shape = [const.Ny, const.Nx, cd]
    with open(file_t, "rb") as file:
        raw = np.fromfile(file, dtype=np.float32, count=np.prod(data_shape), offset=0)
        raw = np.reshape(raw, data_shape)
    return raw[:,:,c]

def read_J(c): return lambda t: read(f"{const.in_dir}/{s}/J_zavg/{str(t // const.Ndts).zfill(4)}", c, 3)
def read_rho(): return lambda t: read(f"{const.in_dir}/{s}/rho_zavg/{str(t // const.Ndts).zfill(4)}", 0, 1)

vmap = np.array([-0.02, +0.02])
vmap_s = {
    "ions": 0.1 * vmap,
    "electrons": vmap,
}
vmap_n = (0, 1)

fig, gs = figure(4, 1)

def plot(i, j, title, vmap, cmap = signed_cmap):
    plot = PlotIm(subplot(fig, gs, i, j), vmap, cmap)
    bx = -const.Lx / 2
    ex = +const.Lx / 2
    by = 0
    ey = const.Ly
    plot.bounds = (bx, ex, by, ey)
    plot.info.set_args(
        title=title,
        xlim=(bx, ex),
        ylim=(by, ey),
        xticks=np.linspace(bx, ex, 5),
        yticks=np.linspace(by, ey, 5),
        xlabel='$x,~c/\\omega_{pe}$',
        ylabel='$y,~c/\\omega_{pe}$',
    )
    return plot

plots = []

def add(i, j, title, vmap, cmap, read):
    plots.append((plot(i, j, title, vmap, cmap), read))

for s in const.sorts:
    add(0, 0, f"$n_{s[0]}$", vmap_n, unsigned_cmap, read_rho()),
    add(1, 0, f"$J_x^{s[0]}$", vmap_s[s], signed_cmap, read_J(0)),
    add(2, 0, f"$J_y^{s[0]}$", vmap_s[s], signed_cmap, read_J(1)),
    add(3, 0, f"$J_z^{s[0]}$", vmap_s[s], signed_cmap, read_J(2)),

    def callback(t):
        for plot, read in plots:
            plot.data = read(t)
            plot.draw()
            plot.axis.plot([const.min_x, const.min_x], [const.min_y, const.max_y], ls="--", c="black")
            plot.axis.plot([const.max_x, const.max_x], [const.min_y, const.max_y], ls="--", c="black")

    process_plots(fig, f"info_{s}", lambda t: f"$t/\\tau = {(t * const.dt / const.tau):.2f}$", np.asarray(plots)[:,0], callback)
    plots.clear()
