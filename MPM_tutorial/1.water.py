import taichi as ti

ti.init(arch=ti.gpu)

########## simulation parameter ##############
particle_num = 512  ##python global variable : not updated in taichi kernel
particle_mass = 1
particle_initial_volume = 1
grid_res = 128
scene_len = 1
grid_dx = scene_len / grid_res
grid_inv_dx= 1/grid_dx
dt = 1e-3

gravity = 9.8

# material property
rho_0 = particle_mass / particle_initial_volume
E = 1
nu = 1
mu_0 = 1
lambda_0 = 1
bulk_modulus = 1
gamma = 7



# taichi data
# particle data
ti_particle_pos = ti.Vector.field(3, ti.f32, particle_num)
ti_particle_vel = ti.Vector.field(3, ti.f32, particle_num)
ti_particle_F = ti.Matrix.field(3, 3, ti.f32, particle_num)
ti_particle_C = ti.Matrix.field(3, 3, ti.f32, particle_num)  # for APIC
ti_particle_Jp = ti.field(ti.f32, particle_num)

# grid data
ti_grid_vel = ti.Vector.field(3, ti.f32, shape=(grid_res, grid_res, grid_res))
ti_grid_mass = ti.field(ti.f32, shape=(grid_res, grid_res, grid_res))


##########################################

@ti.func
def compute_force():
    pass


@ti.kernel
def init():
    pass

@ti.kernel
def init_grid():
    for i, j, k in ti_grid_mass:
        pass

    pass


@ti.kernel
def step():
    pass


if __name__ == '__main__':
    print("hello")
