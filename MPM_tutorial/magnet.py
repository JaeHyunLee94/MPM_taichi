import taichi as ti
from taichi.linalg import sparse_solver
from taichi.linalg import sparse_matrix

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

grid_res = 128
grid_len = 1

grid_dx = grid_len / grid_res
inv_grid_dx = 1 / grid_dx

particle_num = (grid_res) ** 3 / 8

ti_particle_pos = ti.Vector.field(3, ti.f32, particle_num)
ti_particle_vel = ti.Vector.field(3, ti.f32, particle_num)
ti_particle_F = ti.Matrix.field(3, 3, ti.f32, particle_num)
ti_particle_J = ti.field(ti.f32, particle_num)

ti_grid_vel = ti.Vector.field(3, ti.f32, shape=(grid_res, grid_res, grid_res))
ti_grid_H = ti.Vector.field(3, ti.f32, shape=(grid_res, grid_res, grid_res))
ti_grid_M = ti.Vector.field(3, ti.f32, shape=(grid_res, grid_res, grid_res))
ti_grid_mass = ti.field(ti.f32, shape=(grid_res, grid_res, grid_res))
ti_grid_magnetic_potential = ti.field(ti.f32, shape=(grid_res, grid_res, grid_res))

ti_gravity = ti.Vector([0, -9.8, 0])
ti_H_external = ti.Vector([0, 1, 0])


@ti.func
def L(alpha):
    return (ti.exp(2 * alpha) + 1) / (ti.exp(2 * alpha) - 1) - 1 / alpha


@ti.func
def dL(alpha):
    return -(2 / (ti.exp(alpha) - ti.exp(-alpha))) ** 2 + 1 / (alpha ** 2)


@ti.func
def F(phi):
    pass


@ti.func
def dF(phi):
    pass


@ti.func
def PCG():
    pass


@ti.kernel
def substep():
    for p in ti_particle_pos:
        pass
    pass


if __name__ == '__main__':
    print('hello world')
