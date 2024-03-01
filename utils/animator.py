import numpy as np
import os
from functools import partial
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.funcgen import create_oxide_function


def animate_solution_history(filename, oxides, solution_history, y_lim=None):
    """files: list[tuple(x_filename, y_filename)]"""
    data = {}
    t_grid = solution_history.x
    num_oxides = len(oxides)
    y_data = []
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

    def init_func(y_limit):
        ax.set_xlim((0, t_grid.max()))
        if y_limit is None:
            y_limit = max([values.max() for values in data.values()])
        ax.set_ylim((0, y_limit))
        return lines

    def update(frame):
        for j, (new_label, new_y_data) in enumerate(data.items()):
            lines[j].set_data(t_grid, new_y_data[frame])
        return lines

    ani = FuncAnimation(fig, update, n_frames, init_func=partial(init_func, y_limit=y_lim), blit=True)

    ani.save(filename, writer="imagemagick", fps=30)
    plt.close()


def animate_trainable_variables_history(experiment_name, filename, oxides, options):
    t_grid = np.linspace(0, 800, 100)
    num_oxides = len(oxides)
    t_shift = options["t_shift"]

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
        e_var = options["e_var_init"]
        k_var = e_var / oxide["Tm"] + np.log(e_var / oxide["Tm"] ** 2)
        oxide_init = create_oxide_function(k_var, e_var, 10, oxide["Tm"] - t_shift, t_shift=t_shift)
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
        for j, (_, oxide_to_update) in enumerate(oxides.items()):
            e_variable = (record[j + num_oxides]) * options["e_scale"]
            if options["direct_tmax"]:
                t_max_var = record[j] * options["tmax_scale"]
                k_variable = e_variable / t_max_var + np.log(e_variable / t_max_var ** 2)
                t_max_var -= options["t_shift"]
            else:
                k_variable = record[j] * options["k_scale"]
                oxide_ideal_non_normalised = create_oxide_function(k_variable, e_variable, 10,
                                                                   oxide_to_update["Tm"] - t_shift, t_shift=t_shift)
                new_values = oxide_ideal_non_normalised(t_grid)
                t_max_var = t_grid[np.argmax(new_values)]
            oxide_ideal = create_oxide_function(k_variable, e_variable, 10, t_max_var, t_shift=t_shift)
            new_values = oxide_ideal(t_grid)
            lines[j].set_data(t_grid, new_values)
        return lines

    ani = FuncAnimation(fig, update, frames=len(variables_history) - 1, init_func=init_func, blit=True)

    ani.save(filename, writer="imagemagick", fps=30)
    plt.close()
