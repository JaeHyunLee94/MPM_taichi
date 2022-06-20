import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=ti.cuda)

########## simulation parameter ##############
grid_res = 128
particle_num = 2 * (grid_res ** 3) // 4  # 512 * 16  ##python global variable : not updated in taichi kernel
particle_rho = 1

scene_len = 1
grid_dx = scene_len / grid_res
grid_inv_dx = 1 / grid_dx

particle_initial_volume = (grid_dx * 0.5) ** 3
particle_mass = particle_rho * particle_initial_volume
dt = 1e-4

# material property
bulk_modulus = 10  ## lame's second coefficient
gamma = 7  ## compressibility
E = 4

gravity = 9.8
bound = 3

# taichi data
# particle data
ti_particle_pos = ti.Vector.field(3, ti.f32, particle_num)
ti_particle_vel = ti.Vector.field(3, ti.f32, particle_num)
ti_particle_C = ti.Matrix.field(3, 3, ti.f32, particle_num)
ti_particle_Jp = ti.field(ti.f32, particle_num)
#ti_kinetic_energy = ti.field(ti.f32, 100000)

# grid data
ti_grid_vel = ti.Vector.field(3, ti.f32, shape=(grid_res, grid_res, grid_res))
ti_grid_mass = ti.field(ti.f32, shape=(grid_res, grid_res, grid_res))

##########################################

particle_color = (0, 0.5, 1)
particle_radius = 0.01

desired_frame_dt = 1 / 30
ti_frame = ti.field(ti.i32, shape=())

window = ti.ui.Window('Window Title', (1280, 720))
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))


@ti.kernel
def init():
    ti_frame[None] = 0
    # particle initialize
    for p in range(particle_num):
        ti_particle_pos[p] = [
            (ti.random() - 0.5) * 0.5 + 0.5,
            (ti.random() - 0.5) * 0.5 + 0.3,
            (ti.random() - 0.5) * 0.5 + 0.5,
        ]
        ti_particle_Jp[p] = 0.9
        ti_particle_vel[p] = [0, 0, 0]
        ti_particle_C[p] = ti.Matrix.zero(ti.f32, 3, 3)
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

        # pressure = bulk_modulus * ((1 / ti_particle_Jp[p]) ** gamma - 1)
        #
        # stress = -ti.Matrix.identity(ti.f32, 3) * pressure
        # affine = stress                   + particle_mass * ti_particle_C[p]
        # stress = -dt * 4 * E * particle_initial_volume * (ti_particle_Jp[p] - 1) / grid_dx ** 2
        stress = dt * 4 * (bulk_modulus * ((1 / ti_particle_Jp[p]) ** gamma - 1) * ti_particle_Jp[
            p] * particle_initial_volume) / grid_dx ** 2
        affine = ti.Matrix([[stress, 0, 0], [0, stress, 0], [0, 0, stress]]) + particle_mass * ti_particle_C[p]

        # loop unrolling
        # scattering
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight = w[i].x * w[j].y * w[k].z
            dpos = (offset - fx) * grid_dx
            ti_grid_vel[base + offset] += weight * (particle_mass * ti_particle_vel[p] + affine @ dpos)
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

        new_v = ti.zero(ti_particle_vel[p])
        new_C = ti.zero(ti_particle_C[p])

        # gathering
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset - fx) * grid_dx
            weight = w[i].x * w[j].y * w[k].z

            new_v += weight * ti_grid_vel[base + offset]
            new_C += 4 * weight * ti_grid_vel[base + offset].outer_product(dpos) / grid_dx ** 2

        # particle update
        ti_particle_vel[p] = new_v
        ti_particle_C[p] = new_C

        ti_particle_pos[p] += dt * ti_particle_vel[p]
        ti_particle_Jp[p] *= 1 + dt * ti_particle_C[p].trace()


# @ti.kernel
# def calc_energy():
#     K = 0.0
#     for p in ti_particle_pos:
#         K += 0.5 * particle_mass * (
#                     ti_particle_vel[p][0] ** 2 + ti_particle_vel[p][1] ** 2 + ti_particle_vel[p][2] ** 2)
#
#     ti_kinetic_energy[ti_frame[None]] = K

def render_gui():
    global particle_radius
    global particle_color

    global E
    window.GUI.begin("Render setting", 0.02, 0.02, 0.4, 0.15)
    particle_color = window.GUI.color_edit_3("particle color", particle_color)
    particle_radius = window.GUI.slider_float("particle radius", particle_radius, 0.001, 0.1)
    E = window.GUI.slider_float("E", E, 4, 1000)
    if window.GUI.button("restart"):
        init()
    window.GUI.end()

    window.GUI.begin("Simulation setting", 0.02, 0.19, 0.3, 0.1)

    window.GUI.end()


def render():
    camera.track_user_inputs(window, movement_speed=0.01, hold_key=ti.ui.RMB)
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
    print(particle_num)
    while window.running:
        for s in range(int(desired_frame_dt//dt)):
            substep()

        #calc_energy()
        ti_frame[None] += 1

        render()
        render_gui()
        window.show()


    #print(ti_kinetic_energy)
    print("hello")
