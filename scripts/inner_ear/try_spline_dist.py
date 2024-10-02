import imageio.v3 as imageio
import numpy as np
import scipy

import splinebox.basis_functions
import splinebox.spline_curves

from elf.io import open_file
from synaptic_reconstruction.ground_truth.shape_refinement import edge_filter


def get_data():
    center = [105, 710, 510]
    halo = [25, 256, 256]
    bb = tuple(
        slice(ce - ha, ce + ha) for ce, ha in zip(center, halo)
    )
    # bb = np.s_[:]

    tomo_file = "/home/pape/Work/data/moser/em-synapses/Electron-Microscopy-Susi/Analyse/WT strong stim/Mouse 1/modiolar/11/Emb71M1aGridA5sec2.5mod15.rec"  # noqa
    with open_file(tomo_file, "r") as f:
        tomo = f["data"][bb]

    ves_file = "/home/pape/Work/data/moser/em-synapses/Electron-Microscopy-Susi/Analyse/WT strong stim/Mouse 1/modiolar/11/manuell/Vesikel.tif"  # noqa
    vesicles = imageio.imread(ves_file)
    vesicles = vesicles[bb]

    return tomo, vesicles


def _fit_spline(mask, edge):
    # Interpolated edge energy map.
    edge_energy = scipy.interpolate.RectBivariateSpline(
        np.arange(edge.shape[1]), np.arange(edge.shape[0]), edge, kx=2, ky=2, s=1
    )

    def internal_energy(spline, t, alpha, beta):
        return 0.5 * (alpha * spline.eval(t, derivative=1) ** 2 + beta * spline.eval(t, derivative=2) ** 2)

    # Create a circle corresponding to the initial mask.
    coords = np.where(mask)
    offset = [np.mean(coord) for coord in coords]
    # TODO don't hard-code
    radius = 30

    M = 30
    s = np.linspace(0, 2 * np.pi, M + 1)[:-1]
    y = offset[0] + radius * np.sin(s)
    x = offset[1] + radius * np.cos(s)
    knots = np.array([y, x]).T
    initial_knots = knots.copy()

    t = np.linspace(0, M, 400)
    spline = splinebox.spline_curves.Spline(M=M, basis_function=splinebox.basis_functions.B3(), closed=True)
    spline.knots = knots

    alpha = 0
    beta = 0.001

    contours, external_energies = [], []

    def energy_function(control_points, spline, t, alpha, beta):
        control_points = control_points.reshape((spline.M, -1))
        spline.control_points = control_points
        contour = spline.eval(t)
        contours.append(contour.copy())

        # Compute external energy from the edge map
        edge_energy_value = np.sum(edge_energy(contour[:, 0], contour[:, 1], grid=False))
        external_energies.append(-edge_energy_value)

        # Compute internal energy
        internal_energy_value = np.sum(internal_energy(spline, t, alpha, beta))

        # Total energy to minimize
        return -edge_energy_value + internal_energy_value

    initial_control_points = spline.control_points.copy()
    scipy.optimize.minimize(
        energy_function, initial_control_points.flatten(), method="Powell", args=(spline, t, alpha, beta)
    )

    final_knots = spline.eval(np.arange(M))
    return initial_knots, final_knots, spline


def measure_spine_distances(spline1, spline2):
    t = np.linspace(0, spline1.M, 5)
    s = np.linspace(0, spline2.M, 5)

    vals1 = spline1.eval(t)
    vals2 = spline2.eval(s)

    distance_vectors = vals1[:, np.newaxis] - vals2[np.newaxis, :]
    distances = np.linalg.norm(distance_vectors, axis=-1)

    indices = np.unravel_index(np.argmin(distances), distances.shape)
    t_min = t[indices[0]]
    s_min = s[indices[1]]

    def distance(parameters):
        val1 = spline1.eval(parameters[0])
        val2 = spline2.eval(parameters[1])
        return np.linalg.norm(val1 - val2)

    result = scipy.optimize.minimize(distance, np.array([t_min, s_min]), bounds=((0, spline1.M), (0, spline2.M)))
    t_min, s_min = result.x
    return t_min, s_min


def fit_a_spline_2d(tomo, vesicles, pred):
    z = tomo.shape[0] // 2
    x, y, p = tomo[z], vesicles[z], pred[z]

    print("Start splining 1 ...")
    yid = 15
    mask = (y == yid).astype("uint8")
    initial1, spline1, spline_res1 = _fit_spline(mask, p)
    print("Done")

    print("Start splining 2 ...")
    yid = 49
    mask = (y == yid).astype("uint8")
    initial2, spline2, spline_res2 = _fit_spline(mask, p)
    print("Done")

    print("Compute distance ...")
    t_min, s_min = measure_spine_distances(spline_res1, spline_res2)
    print("Done")

    import matplotlib.pyplot as plt
    plt.imshow(p)

    plt.scatter(initial1[:, 1], initial1[:, 0], marker="x", color="black", label="initial knots")
    plt.scatter(spline1[:, 1], spline1[:, 0], marker="o", label="spline")

    plt.scatter(initial2[:, 1], initial2[:, 0], marker="x", color="black", label="initial knots")
    plt.scatter(spline2[:, 1], spline2[:, 0], marker="o", label="spline")

    point1 = spline_res1.eval(t_min)
    point2 = spline_res2.eval(s_min)
    plt.plot([point1[1], point2[1]], [point1[0], point2[0]])

    plt.show()

    # import napari
    # v = napari.Viewer()
    # v.add_image(x)
    # v.add_image(p)
    # v.add_labels(y)
    # v.add_labels(mask)
    # napari.run()


def main():
    tomo, vesicles = get_data()
    pred = edge_filter(tomo, sigma=3.5, method="sobel")

    fit_a_spline_2d(tomo, vesicles, pred)


if __name__ == "__main__":
    main()
