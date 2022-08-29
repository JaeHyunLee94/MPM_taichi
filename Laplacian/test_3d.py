import taichi as ti
import numpy as np

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=ti.cuda)

res = 128
dx = 1 / res
inv_dx = 1 / dx
particle_num = (res ** 3) * 16
bound = 3

ti_particle_pos = ti.Vector.field(3, ti.f32, particle_num)
ti_grid_ours = ti.field(ti.f32, shape=(res, res, res))
ti_grid_mpm = ti.field(ti.f32, shape=(res, res, res))
ti_grid_sol = ti.field(ti.f32, shape=(res, res, res))
ti_grid_fdm = ti.field(ti.f32, shape=(res, res, res))

ti_grid_sol_w_sum1 = ti.field(ti.f32, shape=(res, res, res))
ti_grid_sol_w_sum2 = ti.field(ti.f32, shape=(res, res, res))
ti_grid_sol_n_sum1 = ti.field(ti.f32, shape=(res, res, res))
ti_grid_sol_n_sum2 = ti.field(ti.f32, shape=(res, res, res))


ti_MSE_ours = ti.field(ti.f32, shape=())
ti_MSE_mpm = ti.field(ti.f32, shape=())
ti_MSE_fdm = ti.field(ti.f32, shape=())


@ti.func
def f(x, y, z):
    # analytic solution
    return (x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2 +0.5


@ti.func
def laplacian_f(x, y, z):
    return x+y+z


@ti.kernel
def init():
    ti_MSE_mpm[None] = 0
    ti_MSE_ours[None] = 0

    for p in range(particle_num):
        ti_particle_pos[p] = [
            ti.random(),
            ti.random(),
            ti.random()
        ]

    for i, j, k in ti_grid_sol:
        ti_grid_sol[i, j, k] = laplacian_f(i * dx, j * dx, k * dx)
        ti_grid_mpm[i, j, k] = 0
        ti_grid_ours[i, j, k] = 0
        ti_grid_fdm[i, j, k] = 0


@ti.kernel
def calc_ours():
    for p in range(particle_num):
        Xp = ti_particle_pos[p] * inv_dx
        base = int(Xp)
        fx = Xp - base

        for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
            offset = ti.Vector([i, j, k])
            dist = (fx - offset).norm()
            w1 = (1 - ti.cos(dist * ti.math.pi))
            w2 = (ti.cos(dist * ti.math.pi) - 1)
            ti_grid_sol_w_sum1[base + offset] += w1
            ti_grid_sol_n_sum1[base + offset] += w1 * f(ti_particle_pos[p].x, ti_particle_pos[p].y,
                                                        ti_particle_pos[p].z)
            ti_grid_sol_w_sum2[base + offset] += w2
            ti_grid_sol_n_sum2[base + offset] += w2 * f(ti_particle_pos[p].x, ti_particle_pos[p].y,
                                                        ti_particle_pos[p].z)

    for i, j, k in ti_grid_ours:
        ti_grid_ours[i, j, k] = (ti_grid_sol_n_sum1[i, j, k] / ti_grid_sol_w_sum1[i, j, k] - ti_grid_sol_n_sum2[
            i, j, k] / ti_grid_sol_w_sum2[i, j, k]) * inv_dx * inv_dx


@ti.kernel
def calc_mpm():
    for p in range(particle_num):
        Xp = ti_particle_pos[p] * inv_dx
        base = int(Xp - 1)
        fx = Xp - base
        w = [(1 / 6) * (2 - ti.abs(fx)) ** 3, 0.5 * (ti.abs(fx - 1)) ** 3 - (fx - 1) ** 2 + 2 / 3,
             0.5 * (ti.abs(2 - fx)) ** 3 - (2 - fx) ** 2 + 2 / 3, (1 / 6) * (2 - ti.abs(fx - 3)) ** 3]  # cubic kernel
        ddw = [2 - ti.abs(fx), 3 * ti.abs(fx - 1) - 2, 3 * ti.abs(2 - fx) - 2, 2 - ti.abs(3 - fx)]

        for i, j, k in ti.static(ti.ndrange(4, 4, 4)):
            offset = ti.Vector([i, j, k])
            dd_weight = inv_dx * inv_dx * (
                        ddw[i].x * w[j].y * w[k].z + w[i].x * ddw[j].y * w[k].z + w[i].x * w[j].y * ddw[k].z)
            ti_grid_mpm[base + offset] += dd_weight * f(ti_particle_pos[p].x, ti_particle_pos[p].y,
                                                        ti_particle_pos[p].z)


@ti.kernel
def calc_MSE():
    for i, j, k in ti_grid_sol:
        ti_MSE_mpm[None] += (ti_grid_sol[i, j, k] - ti_grid_mpm[i, j, k]) ** 2
        ti_MSE_ours[None] += (ti_grid_sol[i, j, k] - ti_grid_ours[i, j, k])**2


    ti_MSE_mpm[None] = ti.sqrt(ti_MSE_mpm[None]) / (res ** 3)
    ti_MSE_ours[None] = ti.sqrt(ti_MSE_ours[None]) / (res ** 3)
        #if bound < i < res - bound and bound < j < res - bound and bound < k < res - bound:



@ti.kernel
def calc_FDM():
    for p in range(particle_num):
        Xp = ti_particle_pos[p] * inv_dx
        base = int(Xp - 1)
        fx = Xp - base
        w = [1 / 6 * (2 - fx) ** 3, 0.5 * (fx - 1) ** 3 - (fx - 1) ** 2 + 2 / 3,
             0.5 * (2 - fx) ** 3 - (2 - fx) ** 2 + 2 / 3, 1 / 6 * (fx - 3) ** 3]
    for i, j, k in ti.static(ti.ndrange(4, 4, 4)):
        offset = ti.Vector([i, j, k])
        weight = w[i].x * w[j].y * w[k].z
        ti_grid_fdm[i, j, k] += weight * f(ti_particle_pos[p].x, ti_particle_pos[p].y, ti_particle_pos[p].z)

    for i, j, k in ti_grid_fdm:
        x_p1 = (i + 1) * dx


if __name__ == '__main__':
    init()
    calc_ours()
    calc_mpm()
    calc_MSE()

    print(ti_grid_mpm)
    print('MSE ours:', ti_MSE_ours[None])
    print('MSE mpm:', ti_MSE_mpm[None])
