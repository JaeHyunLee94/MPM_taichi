import taichi as ti
ti.init(arch=ti.cuda)


res =128
dx = 1/res
inv_dx = 1/dx

ti_quantity  = ti.field(ti.f32, shape=(), dtype=ti.f32)
ti_particle_pos = ti.Vector.field(3, ti.f32, 1)
ti_grid_ = ti.field(ti.f32, shape=(res, res, res))


@ti.kernel
def p2g2p():
    pass

if __name__ == '__main__':
    pass