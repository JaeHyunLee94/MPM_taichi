import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

########## simulation parameter ##############
grid_res = 64
particle_num = 10*(grid_res ** 3) // 4  ##python global variable : not updated in taichi kernel
particle_rho = 1

scene_len = 1
grid_dx = scene_len / grid_res
grid_inv_dx = 1 / grid_dx

particle_initial_volume = (grid_dx * 0.5) ** 3
particle_mass = particle_rho * particle_initial_volume
dt = 1e-3

gravity = 9.8
bound = 3
E = 4
# material property
bulk_modulus = 1  ## lame's second coefficient
gamma = 7  ## compressibility

# taichi data
# particle data
ti_particle_pos = ti.Vector.field(3, ti.f32, particle_num)
ti_particle_vel = ti.Vector.field(3, ti.f32, particle_num)
# ti_particle_F = ti.Matrix.field(3, 3, ti.f32, particle_num)  ## need?
ti_particle_Jp = ti.field(ti.f32, particle_num)
ti_particle_vel_grad = ti.Matrix.field(3, 3, ti.f32, particle_num)
# grid data
ti_grid_force = ti.Vector.field(3, ti.f32, shape=(grid_res, grid_res, grid_res))
ti_grid_vel = ti.Vector.field(3, ti.f32, shape=(grid_res, grid_res, grid_res))
ti_grid_mass = ti.field(ti.f32, shape=(grid_res, grid_res, grid_res))

# x2= ti.field(ti.f32)
# t
# ti.root.dense(ti.i, 5).place(x2)

##########################################

particle_color = (0, 0.5, 1)
particle_radius = 0.005

desired_frame_dt = 1 / 60

window = ti.ui.Window('Window Title', (1280, 720))
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))


@ti.kernel
def init():
    # particle initialize
    for p in range(particle_num):
        ti_particle_pos[p] = [
            (ti.random() - 0.5) * 0.5 + 0.5,
            (ti.random() - 0.5) * 0.5 + 0.3,
            (ti.random() - 0.5) * 0.5 + 0.5,
        ]
        ti_particle_Jp[p] = 1
        ti_particle_vel[p] = [0, 0, 0]
        ti_particle_vel_grad[p] = ti.Matrix.zero(ti.f32, 3, 3)
    # grid initialize
    for i, j, k in ti_grid_mass:
        ti_grid_mass[i, j, k] = 0
        ti_grid_vel[i, j, k] = [0, 0, 0]


@ti.kernel
def substep():
    # init grid
    # can be optimized
    for i, j, k in ti_grid_mass:
        ti_grid_mass[i, j, k] = 0
    for i, j, k in ti_grid_vel:
        ti_grid_vel[i, j, k] = [0, 0, 0]


    # p2g
    for p in ti_particle_pos:
        Xp = ti_particle_pos[p] * grid_inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # quadratic kernel
        dw = [fx - 1.5, -2 * (fx - 1), fx - 0.5]

        pressure = bulk_modulus * ((1 / ti_particle_Jp[p]) ** gamma - 1)
        stress = -pressure * ti.Matrix.identity(ti.f32, 3)
        # stress=  -dt*4 * E * particle_initial_volume * (ti_particle_Jp[p] - 1) * ti.Matrix.identity(ti.f32)
        fp = -ti_particle_Jp[p] * particle_initial_volume * stress

        # loop unrolling
        # scattering
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight = w[i].x * w[j].y * w[k].z
            d_weight = ti.Vector([
                grid_inv_dx * dw[i].x * w[j].y * w[k].z,
                grid_inv_dx * w[i].x * dw[j].y * w[k].z,
                grid_inv_dx * w[i].x * w[j].y * dw[k].z,
            ])
            print(weight)
            print(d_weight)
            ti_grid_vel[base + offset] += weight * (particle_mass * ti_particle_vel[p]) + dt*fp @ d_weight
            ti_grid_mass[base + offset] += weight * particle_mass

    # grid update

    for i, j, k in ti_grid_mass:
        if ti_grid_mass[i, j, k] > 0:
            ti_grid_vel[i, j, k] /= ti_grid_mass[i, j, k]
            ti_grid_vel[i, j, k].y -= dt * gravity

        # cond = (I < bound) & (ti_grid_vel[I] < 0) | (I > grid_res - bound) & (ti_grid_vel[I] > 0)
        # ti_grid_vel[I] = ti.select(cond, 0, ti_grid_vel[I])
        if i < bound and ti_grid_vel[i, j, k].x < 0:
            ti_grid_vel[i, j, k].x = 0
        if i > grid_res - bound and ti_grid_vel[i, j, k].x > 0:
            ti_grid_vel[i, j, k].x = 0
        if j < bound and ti_grid_vel[i, j, k].y < 0:
            ti_grid_vel[i, j, k].y = 0
        if j > grid_res - bound and ti_grid_vel[i, j, k].y > 0:
            ti_grid_vel[i, j, k].y = 0
        if k < bound and ti_grid_vel[i, j, k].z < 0:
            ti_grid_vel[i, j, k].z = 0
        if k > grid_res - bound and ti_grid_vel[i, j, k].z > 0:
            ti_grid_vel[i, j, k].z = 0

    # particle update
    for p in ti_particle_pos:
        # gather particle velocity
        Xp = ti_particle_pos[p] * grid_inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # quadratic kernel
        dw = [fx - 1.5, -2 * (fx - 1), fx - 0.5]
        ti_particle_vel[p] = [0, 0, 0]
        ti_particle_vel_grad[p] = ti.Matrix.zero(ti.f32, 3, 3)
        new_v = ti.zero(ti_particle_vel[p])
        new_v_grad = ti.zero(ti_particle_vel_grad[p])

        # gathering
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight = w[i].x * w[j].y * w[k].z
            d_weight = ti.Vector([
                grid_inv_dx * dw[i].x * w[j].y * w[k].z,
                grid_inv_dx * w[i].x * dw[j].y * w[k].z,
                grid_inv_dx * w[i].x * w[j].y * dw[k].z,
            ])
            new_v += weight * ti_grid_vel[base + offset]
            new_v_grad += ti_grid_vel[base + offset].outer_product(d_weight)

        # particle update
        ti_particle_vel[p] = new_v
        ti_particle_vel_grad[p] = new_v_grad
        ti_particle_pos[p] += dt * ti_particle_vel[p]
        ti_particle_Jp[p] *= 1 + dt * ti_particle_vel_grad[p].trace()


def render_gui():
    global particle_radius
    global particle_color

    window.GUI.begin("Render setting", 0.02, 0.02, 0.4, 0.15)
    particle_color = window.GUI.color_edit_3("particle color", particle_color)
    particle_radius = window.GUI.slider_float("particle radius", particle_radius, 0.001, 0.1)
    window.GUI.end()

    window.GUI.begin("Simulation setting", 0.02, 0.19, 0.3, 0.1)

    window.GUI.end()


def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.point_light((0.5, 2, 0.5), (1, 1, 1))

    scene.particles(ti_particle_pos, particle_radius, particle_color)
    canvas.scene(scene)


if __name__ == '__main__':
    init()

    camera.position(2, 2, 2)
    camera.lookat(1, 0.2, 0)
    camera.up(0, 1, 0)
    camera.fov(55)
    camera.projection_mode(ti.ui.ProjectionMode.Perspective)

    while window.running:
        for s in range(int(5)):
            substep()

        render()
        render_gui()
        window.show()

    print("hello")
