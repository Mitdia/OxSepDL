import numpy as np
import os
from deepxde import config
from functools import partial
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.funcgen import create_oxide_function


def animate_from_files(files, filename, y_lim=None):
    """files: list[tuple(x_filename, y_filename)]"""
    data = {}
    for label, (x_file, y_file) in files.items():
        x_data = np.loadtxt(x_file)
        y_data = np.loadtxt(y_file)
        data[label] = y_data

    n_frames = len(y_data)
    t_grid = x_data
    fig, ax = plt.subplots()
    lines = []
    for i, (label, y_data) in enumerate(data.items()):
        lines.append(plt.plot([], [], '-', label=f"{label}")[0])
    plt.legend()
    plt.grid()
    plt.title("Learning process visualisation")
    plt.xlabel("Time")
    plt.ylabel("Oxygen")

    def init_func(ylim):
        ax.set_xlim((0, x_data.max()))
        if ylim is None:
            ylim = max([values.max() for values in data.values()])
        ax.set_ylim((0, ylim))
        return lines

    def update(frame):
        for i, (label, y_data) in enumerate(data.items()):
            lines[i].set_data(t_grid, y_data[frame])
        return lines

    ani = FuncAnimation(fig, update, frames=range(0, n_frames), init_func=partial(init_func, ylim=y_lim), blit=True)

    ani.save(filename, writer="imagemagick", fps=30)
    plt.close()


def animate_solution_history(filename, oxides, solution_history, ylim=None):
    """files: list[tuple(x_filename, y_filename)]"""
    data = {}
    t_grid = solution_history.x
    num_oxides = len(oxides)
    for i, label in enumerate(oxides.keys()):
        y_data = solution_history.y[:, :, i]
        data[label] = y_data
    data["Sum"] = solution_history.y[:, :, num_oxides]
    n_frames = len(y_data)
    fig, ax = plt.subplots()
    lines = []
    for i, (label, y_data) in enumerate(data.items()):
        lines.append(plt.plot([], [], '-', label=f"{label}")[0])
    plt.legend()
    plt.grid()
    plt.title("Learning process visualisation")
    plt.xlabel("Time")
    plt.ylabel("Oxygen")

    def init_func(ylim):
        ax.set_xlim((0, t_grid.max()))
        if ylim is None:
            ylim = max([values.max() for values in data.values()])
        ax.set_ylim((0, ylim))
        return lines

    def update(frame):
        for i, (label, y_data) in enumerate(data.items()):
            lines[i].set_data(t_grid, y_data[frame])
        return lines

    ani = FuncAnimation(fig, update, frames=range(0, n_frames), init_func=partial(init_func, ylim=ylim), blit=True)

    ani.save(filename, writer="imagemagick", fps=30)
    plt.close()


def animate_movie_dumpers(filename="movie.gif", y_lim=None):
    files = {
        "MnO": (f"/kaggle/working/seed_{config.random_seed}_MnO_x.txt",
                f"/kaggle/working/seed_{config.random_seed}_MnO_y.txt"),
        "TiO2": (f"/kaggle/working/seed_{config.random_seed}_TiO2_x.txt",
                 f"/kaggle/working/seed_{config.random_seed}_TiO2_y.txt"),
        "SiO2": (f"/kaggle/working/seed_{config.random_seed}_SiO2_x.txt",
                 f"/kaggle/working/seed_{config.random_seed}_SiO2_y.txt"),
        "Al2O3": (f"/kaggle/working/seed_{config.random_seed}_Al2O3_x.txt",
                  f"/kaggle/working/seed_{config.random_seed}_Al2O3_y.txt"),
        "Sum": (f"/kaggle/working/seed_{config.random_seed}_sum_x.txt",
                f"/kaggle/working/seed_{config.random_seed}_sum_y.txt")
    }
    animate_from_files(files, filename, y_lim)


def animate_trainable_variables_history(experiment_name, filename, oxides, direct_tmax):
    t_grid = np.linspace(0, 800, 100)
    num_oxides = len(oxides)

    def parse_record(rec):
        rec = rec.strip("'")
        rec = rec.replace(",", "")
        rec = rec.replace("[", "")
        rec = rec.replace("]", "")
        return [float(value) for value in rec.split()]

    with open(os.path.join(experiment_name, "variables.dat")) as variables_history_file:
        variables_history = [parse_record(record)[1:] for record in variables_history_file.read().split("\n")][:-2]

    fig, ax = plt.subplots()
    lines = []
    for i, (oxide_name, oxide) in enumerate(oxides.items()):
        e_var = 8e+4
        k_var = e_var / oxide["Tm"] + np.log(e_var / oxide["Tm"] ** 2)
        oxide_init = create_oxide_function(k_var, e_var, 10, oxide["Tm"] - 1400, t_shift=1400)
        values = oxide_init(t_grid)
        plt.plot(t_grid, values, label=f"Init {oxide_name}")
        lines.append(plt.plot(t_grid, values, label=oxide_name)[0])

    plt.legend()
    plt.grid()
    plt.title("Trainable variables history")
    plt.xlabel("Time")
    plt.ylabel("Oxygen")

    def init_func():
        return lines

    def update(frame):
        record = variables_history[frame]
        for i, (oxide_name, oxide) in enumerate(oxides.items()):
            e_var = (record[i + num_oxides]) * 1e+5
            if direct_tmax:
                t_max_var = record[i]
                k_var = e_var / t_max_var + np.log(e_var / t_max_var ** 2)
                t_max_var -= 1400
            else:
                k_var = record[i] * 1e+1
                oxide_ideal_non_normalised = create_oxide_function(k_var, e_var, 10, oxide["Tm"] - 1400, t_shift=1400)
                values = oxide_ideal_non_normalised(t_grid)
                t_max_var = t_grid[np.argmax(values)]
            oxide_ideal = create_oxide_function(k_var, e_var, 10, t_max_var, t_shift=1400)
            values = oxide_ideal(t_grid)
            lines[i].set_data(t_grid, values)
        return lines

    ani = FuncAnimation(fig, update, frames=range(0, len(variables_history) - 1), init_func=init_func, blit=True)

    ani.save(filename, writer="imagemagick", fps=30)
    plt.close()