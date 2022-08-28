import taichi as ti
import numpy as np


arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda


res = 256
dx = 1/res
inv_dx = 1/dx
particle_num = (res ** 3) // 8

ti_particle_pos = ti.Vector.field(3, ti.f32, particle_num)


@ti.kernel
def init():
    pass


@ti.func
def f(x, y, z):
    # analytic solution
    return np.sin(x * y * z)

@ti.func
def laplacian_f(x, y, z):
    return (x**2 + y**2 + z**2) * inv_dx**2 - f(x, y, z)


if __name__ == '__main__':
    pass
