#!/usr/bin/env python3

from plot import *

init_constants("results/ecsimcorr_dx8.0_dt2.0_Np1000/ecsimcorr_dx8.0_dt2.0_Np1000.json")
const.tau = read_scalar(const.config["StepPresets"][3]["tau"])

def avg(d): return sliding_average(d, 100)
    
def read_energy():
    d = np.loadtxt(f"{const.in_dir}/temporal/energy_conservation.txt", skiprows=1)
    return avg(d[:,12]), avg(d[:,13]), avg(d[:,14]), avg(d[:,15])
        
def read_log():
    rm_ni = []
    rm_ne = []
    inj_n = []
    with open(f"{const.in_dir}/output") as f:
        for l in f:
            if "Particles have been removed from \"ions\"" in l:
                rm_ni.append(int(f.readline().split()[3][:-1]))
            elif "Particles have been removed from \"electrons\"" in l:
                rm_ne.append(int(f.readline().split()[3][:-1]))
            elif "Particles have been injected" in l:
                inj_n.append(int(f.readline().split()[3][:-1]))
    return avg(rm_ni), avg(rm_ne), avg(inj_n)

rm_i, rm_e, inj_i, inj_e = read_energy()
rm_ni, rm_ne, inj_n = read_log()

def plot_d(d): return np.arange(len(d)) * const.dt / const.tau, d

fig, gs = figure(1, 2, figsize=(10, 12))

def plot(j):
    plot = PlotLinear(subplot(fig, gs, 0, j))
    plot.info.set_args(
        # xlim=(0, Lt),
        # xticks=np.linspace(0, Lt, 7),
        xlabel="$t / \\tau$"
        # ylim=vmap,
        # yticks=np.linspace(vmap[0], vmap[1], 5),
    )
    return plot

en = plot(0)
n = plot(1)

inj_n_th = 30 * 30 * 5 * 1000 / 86500
inj_i_th = inj_n_th / 1000 * (16 / 511) * 3 / 2
inj_e_th = inj_n_th / 1000 * (2 / 511) * 3 / 2

en.axis.plot(*plot_d(rm_i), label="rm_i")
en.axis.plot(*plot_d(rm_e), label="rm_e")
en.axis.plot(*plot_d(inj_i), label="inj_i")
en.axis.plot(*plot_d(inj_e), label="inj_e")
en.axis.plot([0, len(rm_i) * const.dt / const.tau], [inj_i_th]*2, label="inj_i, th.", ls="--")
en.axis.plot([0, len(rm_i) * const.dt / const.tau], [inj_e_th]*2, label="inj_e, th.", ls="--")
en.axis.legend()
en.info.draw()

n.axis.plot(*plot_d(rm_ni), label="rm_ni")
n.axis.plot(*plot_d(rm_ne), label="rm_ne")
n.axis.plot(*plot_d(inj_n), label="inj_n")
n.axis.plot([0, len(rm_ni) * const.dt / const.tau], [inj_n_th]*2, label="inj_n, th.", ls="--")
n.axis.legend()
n.info.draw()

out_dir = f"{const.out_dir}/other"
makedirs(out_dir)

fig.tight_layout()
fig.savefig(f"{out_dir}/particles_rate.png")