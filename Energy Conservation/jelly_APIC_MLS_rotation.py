import taichi as ti
import matplotlib.pyplot as plt

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=ti.cuda)

########## simulation parameter ##############
grid_res = 64
particle_num = (grid_res ** 3) // 4  # 512 * 16  ##python global variable : not updated in taichi kernel
particle_rho = 1

scene_len = 1
grid_dx = scene_len / grid_res
grid_inv_dx = 1 / grid_dx

particle_initial_volume = (grid_dx * 0.5) ** 3
particle_mass = particle_rho * particle_initial_volume
dt = 1e-4
target_time = 2
# material property

E = 100  # 1000  # Young's modulus
nu = 0.2  # Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
        (1 + nu) * (1 - 2 * nu))  # Lame parameters

gravity = 0  # 9.8
bound = 3
ti_w = ti.Vector([0, 10, 0])
# taichi data
# particle data
ti_particle_pos = ti.Vector.field(3, ti.f32, particle_num)
ti_particle_vel = ti.Vector.field(3, ti.f32, particle_num)
ti_particle_F = ti.Matrix.field(3, 3, ti.f32, particle_num)
ti_particle_C = ti.Matrix.field(3, 3, ti.f32, particle_num)

# grid data
ti_grid_vel = ti.Vector.field(3, ti.f32, shape=(grid_res, grid_res, grid_res))
ti_grid_mass = ti.field(ti.f32, shape=(grid_res, grid_res, grid_res))

ti_total_energy = ti.field(ti.f32, 10000)
ti_kinetic_energy = ti.field(ti.f32, 10000)
ti_strain_energy = ti.field(ti.f32, 10000)
ti_time_elapsed = ti.field(ti.f32, 10000)

ti_frame = ti.field(ti.i32, shape=())
##########################################

particle_color = (1, 0.5, 0)
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
    ti_frame[None] = 0
    ti_time_elapsed[0] = 0
    for p in range(particle_num):
        ti_particle_pos[p] = [
            (ti.random() - 0.5) * 0.3 + 0.5,
            (ti.random() - 0.5) * 0.3 + 0.6,
            (ti.random() - 0.5) * 0.3 + 0.5,
        ]
        # ti_particle_vel[p] = [0,1,0]
        ti_particle_vel[p] = (ti_particle_pos[p] - ti.Vector([0.5, 0.6, 0.5])).cross(ti_w)
        # if ti_particle_pos[p][1] > 0.6:
        #     ti_particle_vel[p] = [0, 2, 0]
        # else:
        #     ti_particle_vel[p] = [0, -2, 0]

        ti_particle_C[p] = ti.Matrix.zero(ti.f32, 3, 3)
        ti_particle_F[p] = ti.Matrix.identity(ti.f32, 3)
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

        U, sig, V = ti.svd(ti_particle_F[p])
        Jp = 1
        R = U @ V.transpose()

        for d in ti.static(range(3)):
            Jp *= sig[d, d]
        stress = -dt * 4 * (particle_initial_volume * (
                2 * mu_0 * (ti_particle_F[p] - R) @ ti_particle_F[p].transpose() + lambda_0 * Jp * (
                Jp - 1) * ti.Matrix.identity(ti.f32, 3))) / grid_dx ** 2
        affine = stress + particle_mass * ti_particle_C[p]

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
            ti_grid_vel[i, j, k] = [0, 0, 0]
        if i > grid_res - bound and ti_grid_vel[i, j, k].x > 0:
            ti_grid_vel[i, j, k] = [0, 0, 0]
        if j < bound and ti_grid_vel[i, j, k].y < 0:
            ti_grid_vel[i, j, k] = [0, 0, 0]
        if j > grid_res - bound and ti_grid_vel[i, j, k].y > 0:
            ti_grid_vel[i, j, k] = [0, 0, 0]
        if k < bound and ti_grid_vel[i, j, k].z < 0:
            ti_grid_vel[i, j, k] = [0, 0, 0]
        if k > grid_res - bound and ti_grid_vel[i, j, k].z > 0:
            ti_grid_vel[i, j, k] = [0, 0, 0]

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
        ti_particle_F[p] = (ti.Matrix.identity(ti.f32, 3) + dt * ti_particle_C[p]) @ ti_particle_F[p]


@ti.kernel
def record():
    K = 0.0
    P = 0.0
    ## energy calculation
    for p in ti_particle_pos:
        K += 0.5 * particle_mass * (
                ti_particle_vel[p][0] ** 2 + ti_particle_vel[p][1] ** 2 + ti_particle_vel[p][2] ** 2)

        U, sig, V = ti.svd(ti_particle_F[p])
        Jp = 1
        R = U @ V.transpose()
        Frobenieus = 0.0
        T = ti_particle_F[p] - R
        # print(T)
        for d in ti.static(range(3)):
            Frobenieus += T[d, 0] * T[d, 0] + T[d, 1] * T[d, 1] + T[d, 2] * T[d, 2]
            Jp *= sig[d, d]

        # print(Frobenieus)
        # print(Jp)
        det = ti_particle_F[p].determinant()
        # print(det)
        P += particle_initial_volume * (
                mu_0 * Frobenieus + 0.5 * lambda_0 * (det - 1) * (det - 1)
        )
    ti_kinetic_energy[ti_frame[None]] = K
    ti_strain_energy[ti_frame[None]] = P
    ti_total_energy[ti_frame[None]] = K + P


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

    elapsed = 0

    substep_num = 5
    while window.running and elapsed < target_time:
        for s in range(substep_num):
            substep()

        record()
        ti_frame[None] += 1
        elapsed += dt * substep_num
        ti_time_elapsed[ti_frame[None]] = elapsed
        render()
        render_gui()
        window.show()

    np_time_elapsed = ti_time_elapsed.to_numpy()
    np_kinetic_energy = ti_kinetic_energy.to_numpy()
    np_strain_energy = ti_strain_energy.to_numpy()
    np_total_energy = ti_total_energy.to_numpy()
    # print(np_time_elapsed[ti_frame[None] - 1])

    plt.plot(np_time_elapsed[:ti_frame[None] - 1], np_kinetic_energy[:ti_frame[None] - 1], label='kinetic energy',
             linewidth=1.0)
    plt.plot(np_time_elapsed[:ti_frame[None] - 1], np_strain_energy[:ti_frame[None] - 1], label='strain energy',
             linewidth=1.0)
    plt.plot(np_time_elapsed[:ti_frame[None] - 1], np_total_energy[:ti_frame[None] - 1], label='total energy',
             linewidth=1.0)

    plt.legend()
    plt.xlabel('Time(s)')
    plt.ylabel('Energy(J)')
    plt.gcf().subplots_adjust(bottom=0.30)
    plt.savefig('rotational secene energy graph')
    plt.show()

    print("program end")
