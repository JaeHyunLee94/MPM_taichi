import taichi as ti

ti.init(arch=ti.cuda)

res = 128
dx = 1 / res
inv_dx = 1 / dx
particle_num = 1

ti_some_quantity = ti.field(ti.f32, particle_num)
ti_particle_pos = ti.Vector.field(3, ti.f32, particle_num)
ti_grid_val = ti.field(ti.f32, shape=(res, res, res))


@ti.kernel
def init():
    for p in range(particle_num):
        ti_some_quantity[p] = 10
        ti_particle_pos[p] = [0.4, 0.4, 0.4]
    for i, j, k in ti_grid_val:
        ti_grid_val[i, j, k] = 0


@ti.kernel
def p2g2p():
    for i, j, k in ti_grid_val:
        ti_grid_val[i, j, k] = 0

    for p in range(particle_num):
        print("before p2g: ", ti_some_quantity[p])
        Xp = ti_particle_pos[p] * inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # quadratic kernel

        # grid_sum2=0
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight = w[i].x * w[j].y * w[k].z
            ti_grid_val[base + offset] += weight * ti_some_quantity[p]
            # grid_sum2+=weight * ti_some_quantity[p]
        # print("after p2g: ", grid_sum2)

    grid_sum = 0.0
    for i, j, k in ti_grid_val:
        grid_sum += ti_grid_val[i, j, k]

    print("after p2g,before g2p: ", grid_sum)
    for p in range(particle_num):
        # gather particle velocity
        Xp = ti_particle_pos[p] * inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # quadratic kernel

        # gathering
        new_p = ti.zero(ti_some_quantity[p])
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight = w[i].x * w[j].y * w[k].z
            new_p += weight * ti_grid_val[base + offset]
        ti_some_quantity[p] = new_p

    for p in range(particle_num):
        print("after g2p: ", ti_some_quantity[p])


if __name__ == '__main__':

    init()
    substep = 5
    for s in range(substep):
        p2g2p()
