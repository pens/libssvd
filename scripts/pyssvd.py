# Copyright 2020 Seth Pendergrass. See LICENSE.
import matplotlib.pyplot as plt
import numpy as np
import ctypes
import imageio

LIBSSVD_PATH = './libssvd/lib/libssvd.so'

def ptr(x):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

class ssvd():
    def __init__(self):
        ctypes.cdll.LoadLibrary(LIBSSVD_PATH)
        self.lib = ctypes.CDLL(LIBSSVD_PATH)

    def svd(self, x, k=10, gpu=False, strm=False):
        self.s = np.ndarray(k, np.float32, order='F')
        self.v = np.ndarray((x.shape[1], k), np.float32, order='F')

        res = self.lib.Svd(ptr(x), x.shape[0], x.shape[1], k, gpu, strm, ptr(self.s), ptr(self.v))
        assert(res == 0)

    def svd_update(self, x):
        res = self.lib.SvdUpdate(ptr(x), 1, ptr(self.s), ptr(self.v))
        assert(res == 0)

    def dmd(self, x, k=10, gpu=False, strm=False):
        self.lmb = np.ndarray(k, np.complex64, order='F')
        self.phi = np.ndarray((x.shape[0], k), np.float32, order='F')

        res = self.lib.Dmd(ptr(x), x.shape[0], x.shape[1], k, gpu, strm, ptr(self.lmb), ptr(self.phi))
        assert(res == 0)

    def dmd_update(self, x):
        res = self.lib.DmdUpdate(ptr(x), 1, ptr(self.lmb), ptr(self.phi))
        assert(res == 0)

    def bgs(self, x, k=10, gpu=False, strm=False):
        self.l = np.ndarray(x.shape, np.complex64, order='F')

        res = self.lib.BackSub(ptr(x), x.shape[0], x.shape[1], k, gpu, strm, ptr(self.l))
        assert(res == 0)

    def bgs_update(self, x):
        res = self.lib.BackSubUpdate(ptr(x), 1, ptr(self.l))
        assert(res == 0)

    def get_elapsed(self):
        self.lib.GetElapsedCalcs.restype = ctypes.c_double
        return self.lib.GetElapsedCalcs()

    def stop(self):
        self.lib.SvdStop()
        self.lib.DmdStop()
        self.lib.BackSubStop()

def svd(x):
    u, s, vt = np.linalg.svd(x, False)
    return u, s, vt.T

def get_face(u, s, v, k):
    return np.clip(u[:, 0:k].dot(np.diag(s)[0:k, 0:k]).dot(v[0, 0:k].T).reshape(192, 168), 0, 255).astype(np.uint8)

def make_image_grid(nrows, ncols, width, height, figure_width=3.9):
    LABEL_SIZE = .4

    aspect = ncols * width / (nrows * height)
    fig_h = (figure_width - LABEL_SIZE) / aspect + LABEL_SIZE

    return plt.subplots(nrows, ncols, True, True, subplot_kw={'frame_on': False}, gridspec_kw={'hspace': 0, 'wspace': 0, 'left': LABEL_SIZE / figure_width, 'right': 1, 'top': 1, 'bottom': LABEL_SIZE / fig_h}, figsize=(figure_width, fig_h))

def get_data_range(x):
    vmin = np.amin(x)
    vmax = np.amax(x)
    vmax = max(np.abs(vmin), np.abs(vmax))
    return -vmax, vmax

def unpack_phi(x, phi_packed, lam):
    m = x.shape[0]
    n = 30
    k = lam.shape[0]

    phi = np.ndarray((m, k), np.complex64)
    i = 0
    while i < k:
        if i < k - 1 and lam[i].conjugate() == lam[i + 1]:
            phi[:, i] = phi_packed[:, i] + 1j * phi_packed[:, i + 1]
            phi[:, i + 1] = phi_packed[:, i] - 1j * phi_packed[:, i + 1]
            i += 2
        else:
            phi[:, i] = phi_packed[:, i]
            i += 1

    b = np.linalg.lstsq(phi, x, rcond=None)[0]
    omega = np.vander(lam, n, True)
    dynamics = np.diag(b).dot(omega)

    return phi, b, omega, dynamics


def reconstruct(x, phi_packed, lam):
    phi, _, _, dynamics = unpack_phi(x, phi_packed, lam)

    return phi.dot(dynamics).real

def get_mode(modes, i):
    return modes[:, i].reshape(199, 449, order='F').real


def make_recon(amps, amps_ctrl, eigs, eigs_ctrl, modes, modes_ctrl, run, i):
    b = (amps if run == 0 else amps_ctrl)[0:i]
    lam = (eigs if run == 0 else eigs_ctrl)[0:i]
    omega = np.vander(lam, 30, True)

    dynamics = np.diag(b).dot(omega)[0:i, :]
    phi = (modes if run == 0 else modes_ctrl)[:, 0:i]
    return phi.dot(dynamics).real[:, -1].reshape(199, 449, order='F')


def load_matrix(input_dir, res):
    start = 200
    stop = 321
    in_file = input_dir + 'pevid/Walking_day_indoor_4_original.mp4'

    r = imageio.get_reader(in_file, size=res)
    r.set_image_index(start)
    x = np.zeros((res[0] * res[1], stop - start), np.float32, order='F')
    for i in range(stop - start):
        x[:, i] = r.get_next_data()[:, :, 0].flatten()# / 255.0

    return x


def unpack_modes(phi_packed, lam):
    k = lam.shape[0]

    phi = np.ndarray(phi_packed.shape, np.complex64)
    i = 0
    while i < k:
        if i < k - 1 and lam[i].conjugate() == lam[i + 1]:
            phi[:, i] = phi_packed[:, i] + 1j * phi_packed[:, i + 1]
            phi[:, i + 1] = phi_packed[:, i] - 1j * phi_packed[:, i + 1]
            i += 2
        else:
            phi[:, i] = phi_packed[:, i]
            i += 1

    return phi

def sort_eigs(eigs):
    return np.argsort(np.abs(np.log(eigs)))
