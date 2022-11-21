import taichi as ti
import matplotlib.pyplot as plt

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=ti.cuda)

########## simulation parameter ##############
grid_res = 256
particle_num = 1  # (grid_res ** 3) // 4  # 512 * 16  ##python global variable : not updated in taichi kernel
particle_rho = 1

scene_len = 1
grid_dx = scene_len / grid_res
grid_inv_dx = 1 / grid_dx

# particle_initial_volume = (grid_dx * 0.5) ** 3
particle_mass = 3  # particle_rho * particle_initial_volume
dt = 5e-3
# material property


gravity = 0  # 9.8
bound = 3
target_frame = 50
# taichi data
# particle data
ti_particle_pos = ti.Vector.field(3, ti.f32, particle_num)
ti_particle_vel = ti.Vector.field(3, ti.f32, particle_num)

# grid data
ti_grid_vel = ti.Vector.field(3, ti.f32, shape=(grid_res, grid_res, grid_res))
ti_grid_mass = ti.field(ti.f32, shape=(grid_res, grid_res, grid_res))

ti_particle_energy = ti.field(ti.f32, target_frame)
ti_grid_energy = ti.field(ti.f32, target_frame)
ti_particle_momentum = ti.Vector.field(3, ti.f32, target_frame)
ti_grid_momentum = ti.Vector.field(3, ti.f32, target_frame)
ti_grid_mass_sum = ti.field(ti.f32, target_frame)

ti_collision_marker = ti.field(ti.i32, target_frame)

ti_frame = ti.field(ti.i32, shape=())
##########################################

particle_color = (1, 0.5, 0)
particle_radius = 0.005

window = ti.ui.Window('Window Title', (1280, 720))
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))


@ti.kernel
def init():
    # particle initialize
    ti_frame[None] = 0
    for i in range(target_frame):
        ti_particle_energy[i] = 0
        ti_grid_energy[i] = 0
    for p in range(particle_num):
        ti_particle_pos[p] = [
            (ti.random() - 0.5) * 0.2 + 0.5,
            (ti.random() - 0.5) * 0.2 + 0.5,
            (ti.random() - 0.5) * 0.2 + 0.5,
        ]
        # ti_particle_pos[p] = [0.478, 0.521, 0.5111]
        ti_particle_vel[p] = [1, 1, 1]

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

        # print(particle_mass * (ti_particle_vel[p].norm_sqr()) / 2)
        ti_particle_energy[ti_frame[None]] += particle_mass * (ti_particle_vel[p].norm_sqr())
        ti_particle_momentum[ti_frame[None]] += particle_mass * ti_particle_vel[p]
        # print(ti_frame[None],particle_mass * ti_particle_vel[p].norm_sqr() / 2)
        # scattering
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight = w[i].x * w[j].y * w[k].z
            ti_grid_vel[base + offset] += weight * (particle_mass * ti_particle_vel[p])
            ti_grid_mass[base + offset] += weight * particle_mass
            # ti_grid_energy[ti_frame[None]] += ti_grid_vel[base + offset].norm_sqr() / 2 * ti_grid_mass[base + offset]

            # grid update

    for i, j, k in ti_grid_mass:
        if ti_grid_mass[i, j, k] > 0:
            ti_grid_energy[ti_frame[None]] += ti_grid_vel[i, j, k].norm_sqr() / ti_grid_mass[i, j, k]
            ti_grid_momentum[ti_frame[None]] += ti_grid_vel[i, j, k]
            ti_grid_mass_sum[ti_frame[None]] += ti_grid_mass[i, j, k]
            ti_grid_vel[i, j, k] /= ti_grid_mass[i, j, k]
            # ti_grid_vel[i, j, k].y -= dt * gravity
            # print(ti_grid_mass[i, j, k] * (ti_grid_vel[i, j, k].norm_sqr()) / 2)
            # ti_grid_energy[ti_frame[None]] += ti_grid_mass[i, j, k] * (ti_grid_vel[i, j, k].norm_sqr()) / 2

        # cond = (I < bound) & (ti_grid_vel[I] < 0) | (I > grid_res - bound) & (ti_grid_vel[I] > 0)
        # ti_grid_vel[I] = ti.select(cond, 0, ti_grid_vel[I])

        if i < bound and ti_grid_vel[i, j, k].x < 0:
            ti_grid_vel[i, j, k] = [0, 0, 0]
            ti_collision_marker[ti_frame[None]] += 1
        if i > grid_res - bound and ti_grid_vel[i, j, k].x > 0:
            ti_grid_vel[i, j, k] = [0, 0, 0]
            ti_collision_marker[ti_frame[None]] += 1
        if j < bound and ti_grid_vel[i, j, k].y < 0:
            ti_grid_vel[i, j, k] = [0, 0, 0]
            ti_collision_marker[ti_frame[None]] += 1
        if j > grid_res - bound and ti_grid_vel[i, j, k].y > 0:
            ti_grid_vel[i, j, k] = [0, 0, 0]
            ti_collision_marker[ti_frame[None]] += 1
        if k < bound and ti_grid_vel[i, j, k].z < 0:
            ti_grid_vel[i, j, k] = [0, 0, 0]
            ti_collision_marker[ti_frame[None]] += 1
        if k > grid_res - bound and ti_grid_vel[i, j, k].z > 0:
            ti_grid_vel[i, j, k] = [0, 0, 0]
            ti_collision_marker[ti_frame[None]] += 1

    # particle update
    for p in ti_particle_pos:
        # gather particle velocity
        Xp = ti_particle_pos[p] * grid_inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # quadratic kernel

        new_v = ti.zero(ti_particle_vel[p])

        # gathering
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight = w[i].x * w[j].y * w[k].z
            new_v += weight * ti_grid_vel[base + offset]

        # particle update
        ti_particle_vel[p] = new_v
        ti_particle_pos[p] += dt * ti_particle_vel[p]


def render_gui():
    global particle_radius
    global particle_color

    # global E
    window.GUI.begin("setting", 0.02, 0.02, 0.4, 0.15)
    particle_color = window.GUI.color_edit_3("particle color", particle_color)
    particle_radius = window.GUI.slider_float("particle radius", particle_radius, 0.001, 0.1)
    if window.GUI.button("restart"):
        init()
    window.GUI.end()

    # window.GUI.begin("Simulation setting", 0.02, 0.19, 0.3, 0.1)
    #
    # window.GUI.end()


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

    while window.running and ti_frame[None] < target_frame:
        substep()
        render()
        render_gui()
        window.show()
        ti_frame[None] += 1

    for i in range(0, target_frame):
        # print(i, ti_particle_energy[i], ti_grid_energy[i], ti_particle_energy[i] <= ti_grid_energy[i],
        #       ti_collision_marker[i])
        print(i, ti_particle_momentum[i], ti_grid_momentum[i], ti_grid_mass_sum[i])
