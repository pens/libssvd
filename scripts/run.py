# Copyright 2020 Seth Pendergrass. See LICENSE.
# %% Setup
import os
import ctypes
import glob
import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
import pydmd
import scipy.io
from matplotlib.animation import FuncAnimation, ImageMagickWriter
from netCDF4 import Dataset
os.chdir(os.path.dirname(__file__))
from pyssvd import *
os.chdir(os.path.dirname(__file__) + '/../..')

INPUT_DIR = './data/'
OUTPUT_DIR = './results/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global Matplotlib Configuration
colors = scipy.io.loadmat(INPUT_DIR + 'CCcool.mat')['CC']
cmap = mpl.colors.ListedColormap(colors)
plt.register_cmap('custom', cmap)
plt.style.use('ggplot')
mpl.rcParams.update({
    'axes.labelsize': 'small',
    'legend.fontsize': 'small',
    'image.cmap': 'custom',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small'
})

lib = ssvd()

# %% SVD Yale Faces
h = 192
w = 168
k = 65

x = np.ndarray((h * w, k), np.float32, order='F')
for i, face in enumerate(glob.glob(INPUT_DIR + '/yaleB01/*.pgm')):
    x[:, i] = imageio.imread(face).flatten()

# Numpy
u_ctrl, s_ctrl, v_ctrl = svd(x)

# Ours
lib.svd(x, k, True)
s = lib.s
v = lib.v
u = x.dot(v).dot(np.diag(np.reciprocal(s)))

# %% SVD Yale Faces Singular Values
fig, ax = plt.subplots(2, 1, True, gridspec_kw={'hspace': 0, 'right': 1, 'top': 1}, figsize=(3.9, 3.9))
ax[0].set_ylabel(r'$\sigma_k (\times10^5)$')
ax[1].set_ylabel(r'$\log \sigma_k$')
ax[1].set_xlabel('$k$')
ax[0].ticklabel_format(axis='y', style='sci', scilimits=(5, 5), useMathText=True)
fig.align_ylabels(ax[:])

ax[0].plot(s_ctrl, 'rx', s, 'b+')
ax[1].plot(np.log(s_ctrl), 'rx', np.log(s), 'b+')
ax[0].legend(['SVD', 'MoS'])

plt.savefig(OUTPUT_DIR + 'svd_faces_sing_vals.pdf')

# %% SVD Yale Faces Recon
fig, ax = make_image_grid(2, 3, 168, 192)
ax[0, 0].set_ylabel('SVD')
ax[1, 0].set_ylabel('MoS')
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 0].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

for i, k in enumerate([1, 32, 65]):
    ax[0, i].imshow(get_face(u_ctrl, s_ctrl, v_ctrl, k), cmap='gray', aspect='auto')
    ax[1, i].imshow(get_face(u, s, v, k), cmap='gray', aspect='auto')
    ax[1, i].set_xlabel('k={}'.format(k))

plt.savefig(OUTPUT_DIR + 'svd_faces_recon.pdf')

# %% DMD Cylinder
n = 30
k = 10
h = 199
w = 449
COMPARISON_IDXS = [1, n // 2, n]

x = scipy.io.loadmat(INPUT_DIR + '/CYLINDER_ALL.mat')['VORTALL'].astype(np.float32)

# Inputs & PyDMD
orig = []
recon_ctrl = []
dmd = pydmd.DMD(k, 0, True)
for i in COMPARISON_IDXS:
    orig.append(x[:, i].reshape(h, w, order='F'))
    dmd.fit(x[:, i:n+i])
    recon_ctrl.append(dmd.reconstructed_data.real[:, -1].reshape(h, w, order='F'))

idxs_ctrl = sort_eigs(dmd.eigs)
eigs_ctrl = dmd.eigs[idxs_ctrl]
modes_ctrl = dmd.modes[:, idxs_ctrl]
amps_ctrl = dmd._b[idxs_ctrl]

# Ours
recon = []
lib.dmd(x[:, :n], k, True, True)
for i in range(1, 1 + n):
    lib.dmd_update(x[:, i:i+n])
    if i in COMPARISON_IDXS:
        recon.append(reconstruct(x[:, i], lib.phi, lib.lmb)[:, -1].reshape(h, w, order='F'))
lib.stop()

phi, b = unpack_phi(x[:, 0], lib.phi, lib.lmb)[0:2]
idxs = sort_eigs(lib.lmb)
eigs = lib.lmb[idxs]
modes = phi[:, idxs]
amps = b[idxs]

if np.sign(amps[0]) == -np.sign(amps_ctrl[0]):
    amps = -amps
    modes = -modes

# %% DMD Cylinder Streaming
fig, ax = make_image_grid(3, 3, w, h, 8)
ax[0, 0].set_ylabel('Original')
ax[1, 0].set_ylabel('PyDMD')
ax[2, 0].set_ylabel('Streaming DMD')
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])

for i, frame in enumerate(COMPARISON_IDXS):
    for j, idx in enumerate([orig, recon_ctrl, recon]):
        img = idx[i]
        vmin, vmax = get_data_range(img)
        ax[j, i].imshow(img, aspect='auto', vmin=vmin, vmax=vmax)
        ax[j, i].contour(img, np.arange(-5, 5.5, .5), colors='black', linewidths=.5)
        ax[j, i].add_artist(plt.Circle((49, 99), 25, color='black', zorder=10))
    ax[2, i].set_xlabel('Frame {}'.format(frame))

plt.savefig(OUTPUT_DIR + 'dmd_cylinder_streaming.pdf')

# %% DMD Cylinder Eigs
fig, ax = plt.subplots(gridspec_kw={'top': 1, 'right': 1, 'left': 0}, figsize=(3.9, 3.9))
ax.set_aspect(1)
ax.set_xlabel('Real')
ax.set_xlim(.1, 1.1)
ax.set_xticks([.5, 1])
ax.set_ylabel('Imag')
ax.set_ylim(-1.1, 1.1)
ax.set_yticks([-1, 0, 1])

ax.plot(eigs_ctrl.real, eigs_ctrl.imag, 'rx', eigs.real, eigs.imag, 'b+')
ax.add_artist(plt.Circle((0, 0), 1, color='black', fill=False, linewidth=.5))
ax.legend(['Static', 'Streaming'])

plt.savefig(OUTPUT_DIR + 'dmd_cylinder_eigs.pdf')

# %% DMD Cylinder Modes
fig, ax = make_image_grid(2, 6, w, h, 8)
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 0].set_ylabel('Static')
ax[1, 0].set_ylabel('Streaming')

for i in range(6):
    for j, m in enumerate([modes, modes_ctrl]):
        img = get_mode(m, i)
        vmin, vmax = get_data_range(img)
        ax[j, i].imshow(img, aspect='auto', vmin=vmin, vmax=vmax)
        ax[j, i].contour(img, np.arange(-.01, .01, .0025), colors='black', linewidths=.5)
        ax[j, i].add_artist(plt.Circle((49, 99), 25, color='black', zorder=10))
    ax[1, i].set_xlabel('Mode {}'.format(j))

plt.savefig(OUTPUT_DIR + 'dmd_cylinder_modes.pdf')

# %% DMD Cylinder Reconstruction
fig, ax = make_image_grid(2, 3, w, h, 8)
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 0].set_ylabel('Static')
ax[1, 0].set_ylabel('Streaming')
ax[1, 0].set_xlabel('Modes 0-1')
ax[1, 1].set_xlabel('Modes 0-5')
ax[1, 2].set_xlabel('All Modes')

for i in range(2):
    for j, idx in enumerate([1, 5, 10]):
        img = make_recon(amps, amps_ctrl, eigs, eigs_ctrl, modes, modes_ctrl, i, idx)
        vmin, vmax = get_data_range(img)
        ax[i, j].imshow(img, aspect='auto', vmin=vmin, vmax=vmax)
        ax[i, j].contour(img, np.arange(-5, 5.5, .5), colors='black', linewidths=.5)
        ax[i, j].add_artist(plt.Circle((49, 99), 25, color='black', zorder=10))

plt.savefig(OUTPUT_DIR + 'dmd_cylinder_recon.pdf')

# %% DMD Cylinder Original
fig, ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Original')

img = x[:, 59].reshape(h, w, order='F')
vmin, vmax = get_data_range(img)
ax.imshow(img, vmin=vmin, vmax=vmax)
ax.contour(img, np.arange(-5, 5.5, .5), colors='black', linewidths=.5)
ax.add_artist(plt.Circle((49, 99), 25, color='black', zorder=10))

plt.savefig(OUTPUT_DIR + 'dmd_cylinder_orig.pdf')

# %% DMD Cylinder Modes .gif
fig, ax = plt.subplots(figsize=(9.6, 8.4))
ax.set_xticks([])
ax.set_yticks([])
im = ax.imshow(get_mode(modes, 0), animated=True)

def update(frame):
    img = get_mode(modes, frame)
    vmin, vmax = get_data_range(img)

    im.set_clim(vmin, vmax)
    im.set_array(img)
    ax.collections = []
    ax.contour(img, np.arange(-.01, .01, .0025), colors='black', linewidths=1)
    ax.add_artist(plt.Circle((49, 99), 25, color='black', zorder=10))
    return im,

ani = FuncAnimation(fig, update, modes.shape[1], blit=True)
ani.save(OUTPUT_DIR + 'dmd_cylinder_modes.gif', ImageMagickWriter())

# %% Benchmarks PEViD
NUM_WARMUPS = 3
NUM_RUNS = 3
NUM_STEPS = 3
resolutions = [(3840, 2160), (2560, 1440), (1920, 1080), (1280, 720), (960, 540), (640, 360)]
widths = [120, 100, 80, 60, 40, 20]
res_fixed = (1920, 1080)
width_fixed = 80

fns = [lib.svd, lib.dmd, lib.bgs]
fns_update = [lib.svd_update, lib.dmd_update, lib.bgs_update]
idx = pd.MultiIndex.from_product([['svd', 'dmd', 'bgs'], ['cpu', 'gpu'], ['stat', 'strm'], range(NUM_RUNS + 1)])

def run_benchmark(x, times):
    def run_iteration(fn_idx, is_gpu):
        fns[fn_idx](x[:, :-1], gpu=is_gpu, strm=True)
        elapsed_static = lib.get_elapsed()

        elapsed = 0
        for i in range(NUM_STEPS):
            fns_update[fn_idx](x[:, i+1:])
            elapsed += lib.get_elapsed()
        lib.stop()
        return elapsed_static, elapsed / NUM_STEPS

    for proc in ['cpu', 'gpu']:
        for fn_idx, alg in enumerate(['svd', 'dmd', 'bgs']):
            for run in range(NUM_WARMUPS + NUM_RUNS):
                elapsed = run_iteration(fn_idx, proc == 'gpu')

                if run >= NUM_WARMUPS:
                    print('{} {} {} {} {}'.format(alg, proc, run - NUM_WARMUPS + 1, elapsed[0], elapsed[1]))
                    r = run - NUM_WARMUPS + 1
                    times[alg, proc, 'stat'][r] = elapsed[0]
                    times[alg, proc, 'strm'][r] = elapsed[1]
                    # Index 0 is the mean
                    times[alg, proc, 'stat'][0] += elapsed[0]
                    times[alg, proc, 'strm'][0] += elapsed[1]

            times[alg, proc, 'stat'][0] /= NUM_RUNS
            times[alg, proc, 'strm'][0] /= NUM_RUNS

#%% Benchmarks PEViD Resolution
print("\nVarying Resolution")
times_res = pd.DataFrame(np.zeros((len(idx), len(resolutions))), idx, resolutions)
for res in resolutions:
    print('{}'.format(res))
    run_benchmark(load_matrix(INPUT_DIR, res)[:, :width_fixed+1], times_res[res])
times_res.to_csv(OUTPUT_DIR + 'benchmark_resolutions.csv')

##%% Benchmnarks PEViD Width
#print("\nVarying width")
#times_width = pd.DataFrame(np.zeros((len(idx), len(widths))), idx, widths)
#x = load_matrix(INPUT_DIR, res_fixed) # loads all 121 so only need to do once
#for width in widths:
#    print('{}'.format(width))
#    run_benchmark(x[:, :width+1], times_width[width])
#times_width.to_csv(OUTPUT_DIR + 'width.csv')

# %% Benchmarks PEViD Relative
fig, ax = plt.subplots(3, 1, 'col', 'row', gridspec_kw={'hspace': 0, 'right': 1, 'top': 1, 'bottom': .6 / 3.9}, figsize=(3.9, 3.9))
ax[0].set_ylabel('SVD (s)')
ax[1].set_ylabel('DMD (s)')
ax[2].set_ylabel('Bg Sub (s)')
ax[2].set_xlabel('Height (16:9 Aspect Ratio)')
xtick_labels = [360, 540, 720, 1080, 1440, 2160]
xticks = [i * (i // 9) * 16 for i in xtick_labels]
ax[0].set_xticks(xticks)
ax[2].set_xticklabels(xtick_labels, rotation='vertical')
fig.align_ylabels(ax[:])

times = pd.read_csv(OUTPUT_DIR + 'benchmark_resolutions.csv', index_col=[0, 1, 2, 3])

for i, lbl in enumerate(['svd', 'dmd', 'bgs']):
    time_cpu = times.loc[lbl, 'cpu', 'stat'].iloc[0][::-1]
    for strm in ['stat', 'strm']:
        for proc in ['cpu', 'gpu']:
            if strm == 'stat' and proc == 'cpu':
                continue
            time = times.loc[lbl, proc, strm].iloc[0][::-1]
            ax[i].plot(xticks, time_cpu / time)
ax[0].legend(['Streaming CPU', 'GPU', 'Streaming GPU'], loc='upper left')
plt.savefig(OUTPUT_DIR + 'benchmark_resolutions.pdf')

# %% DMD NOAA
m = 691150 #1440 * 720
n = 365 // 5
k = 20

x = np.ndarray((m, n), np.float32, order='F')
sst = Dataset('data/noaa/sst.day.mean.2018.v2.nc').variables['sst'][:]
for i in range(n):
    x[:, i] = sst[i * 5].compressed()

# PyDMD
dmd = pydmd.DMD(k, 0, True)
dmd.fit(x)
idxs2 = sort_eigs(dmd.eigs)

# Ours
lib.dmd(x, k)
idxs = sort_eigs(lib.lmb)
phi = unpack_modes(lib.phi, lib.lmb)

# Make signs consistent
for i in range(phi.shape[1]):
    if np.sign(dmd.modes[0, i]) == -np.sign(phi[0, i]):
        phi[:, i] = -phi[:, i]

# %% DMD NOAA Modes
fig, ax = make_image_grid(2, 6, 1440, 720, 8)
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 0].set_ylabel('Static')
ax[1, 0].set_ylabel('Streaming')
palette = mpl.colors.ListedColormap((0, 0, 0, 0))
palette.set_bad((0, .5, 0), alpha=1.0)

x = sst[0, :, :].copy()
for i in range(6):
    x[~x.mask] = phi[:, idxs[i]].real
    vmin, vmax = get_data_range(x[::-1, :])
    ax[0, i].imshow(x[::-1, :], aspect='auto', vmin=vmin, vmax=vmax)
    ax[0, i].imshow(x[::-1, :], cmap=palette)

    x[~x.mask] = dmd.modes[:, idxs2[i]].real
    vmin, vmax = get_data_range(x[::-1, :])
    ax[1, i].imshow(x[::-1, :], aspect='auto', vmin=vmin, vmax=vmax)
    ax[1, i].imshow(x[::-1, :], cmap=palette)

    ax[1, i].set_xlabel('Mode {}'.format(i))

# %% DMD NOAA Benchmark
N = 60
k = 20
x = np.ndarray((691150, N), np.float32, order='F')

idx = pd.MultiIndex.from_product([range(1981, 2020), range(366)])
data = pd.DataFrame(np.zeros((len(idx), 2)), idx, ['cpu', 'gpu'])

for gpu in [False, True]:
    lbl = 'gpu' if gpu else 'cpu'

    print('{}'.format(lbl))
    sst = Dataset('data/noaa/sst.day.mean.1981.v2.nc').variables['sst'][:]
    for i in range(N):
        x[:, i] = sst[i].compressed()

    start = 365 - sst.shape[0]

    lib.dmd(x, k, gpu, True)
    elapsed = lib.get_elapsed()
    data[lbl][1981, start + N - 1] = lib.get_elapsed()

    for d in range(N, sst.shape[0]):
        x[:, :-1] = x[:, 1:]
        x[:, -1] = sst[d].compressed()
        lib.dmd_update(x)
        elapsed += lib.get_elapsed()
        data[lbl][1981, start + d] = lib.get_elapsed()

    for y in range(1982, 2020):
        print('{} {}'.format(y - 1, elapsed))
        sst = Dataset('data/noaa/sst.day.mean.{}.v2.nc'.format(y)).variables['sst'][:]

        for d in range(sst.shape[0]):
            x[:, :-1] = x[:, 1:]
            x[:, -1] = sst[d].compressed()
            lib.dmd_update(x)
            elapsed += lib.get_elapsed()
            data[lbl][y, d] = lib.get_elapsed()

    print('2019 {}'.format(elapsed))

mean_idx = pd.MultiIndex.from_tuples([(0, 0)])
mean = pd.DataFrame(np.zeros((1, 2)), mean_idx, ['cpu', 'gpu'])
sums = data.sum()
mean['cpu'] = sums['cpu']
mean['gpu'] = sums['gpu']
data_out = pd.concat([mean, data])
data_out.to_csv(OUTPUT_DIR + 'benchmark_NOAA.csv')

# %%
