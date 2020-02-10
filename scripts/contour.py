# Copyright 2020 Seth Pendergrass. See LICENSE.
from mayavi import mlab

x = scipy.io.loadmat(INPUT_DIR + '/CYLINDER_ALL.mat')['VORTALL'].astype(np.float32)

x3d = np.ndarray((199, 449, x.shape[1]))
for i in range(x.shape[1]):
    x3d[:, :, i] = x[:, i].reshape(199, 449, order='F').real

colormap = np.ndarray((256, 4))
for i in range(64):
    r = 4 * i
    colormap[r:r+4, 0:3] = 255 * colors[i]
    colormap[r:r+4, 3] = 128

obj = mlab.contour3d(x3d, contours=20, transparent=True)
obj.module_manager.scalar_lut_manager.lut.table = colormap
mlab.draw()
mlab.savefig(OUTPUT_DIR + 'contour.png')
#mlab.show()
