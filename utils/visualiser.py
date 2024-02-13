import numpy as np
import deepxde as dde
import torch
import os
from deepxde import config
from matplotlib import pyplot as plt
from cycler import cycler
from utils.preprocessing import find_region_of_main_peak
from utils.funcgen import create_oxide_function


monochrome_settings = cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':']) * cycler('marker', ['^', ',', '.'])


def plot_loss_history(loss_history, verbose=False, filename="LossHistory.png"):
    plt.plot(loss_history.steps, [loss.sum() for loss in loss_history.loss_train], label="train loss")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative loss")
    plt.title("Loss history")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    if verbose:
        plt.show()


def plot_result(model, oxides, references, t_shift, verbose=False, filename="Result.png", monochrome=False):
    num_oxides = len(oxides)
    t_grid = dde.geometry.Interval(0, 800).uniform_points(1000, False)
    result = model.predict(t_grid)
    fig, axs = plt.subplots(figsize=(10, 6))
    if monochrome:
        axs.set_prop_cycle(monochrome_settings)
    for i, oxide_name in enumerate(oxides):
        plt.plot(t_grid, result[:, i], ".", label=f"{oxide_name}")
    predicted_sum = result[:, num_oxides]
    plt.plot(t_grid, predicted_sum, ".", label=f"summ")
    for i, (reference, temp) in enumerate(references):
        shifted_temp = temp - t_shift
        predicted_sum = model.predict(shifted_temp.reshape(-1, 1))[:, num_oxides]
        error = np.mean((reference - predicted_sum) ** 2)
        plt.plot(shifted_temp[::1], reference[::1], label=f"Reference {i + 1}: loss {error:.4}")
    plt.legend()
    plt.grid()

    plt.title(f"Predicted peaks and references. Seed: {dde.config.random_seed}.")
    plt.xlabel("Time")
    plt.ylabel("Oxygen")
    plt.savefig(filename)
    if verbose:
        plt.show()


def print_final_losses(loss_history, num_oxides, num_references):
    mean_ode_loss = np.mean(loss_history.loss_test[-1][:num_oxides]) / 1e+5
    mean_tbeg_loss = np.mean(loss_history.loss_test[-1][num_oxides:2 * num_oxides]) / 1e+3
    mean_ref_loss = np.mean(loss_history.loss_test[-1][2 * num_oxides::2]) / 5e+0 * num_references
    mean_mpeak_loss = np.mean(loss_history.loss_test[-1][2 * num_oxides + 1::2]) / 1e+2 * num_references
    print(
        f"Predicted peaks and references. "
        f"Seed: {dde.config.random_seed}. "
        f"Mean ode loss: {mean_ode_loss:.2}. "
        f"Mean tbeg loss: {mean_tbeg_loss:.2}. "
        f"Mean ref loss {mean_ref_loss:.2}. "
        f"Mean mpeak loss {mean_mpeak_loss:.2}."
    )


def plot_ode_residual(model, oxide_params, verbose=False, filename="ODELoss.png"):
    num_oxides = len(oxide_params)
    t_grid = dde.geometry.Interval(0, 800).uniform_points(2000, False)
    full_error = model.predict(t_grid, operator=model.data.pde)
    squared_full_error = [error ** 2 for error in full_error]
    average_pde_error = np.sqrt(np.mean(squared_full_error[:num_oxides], axis=0))
    points = model.data.geom.random_points(200, "residual-based")
    plt.figure(figsize=(20, 8))
    for i, oxide_name in enumerate(oxide_params):
        plt.plot(t_grid, np.abs(full_error[i]), label=f"{oxide_name} loss")

    plt.plot(t_grid, average_pde_error, ".", label=f"average pde error")
    plt.plot(points, np.zeros_like(points), "x", label=f"chosen points")
    plt.legend()
    plt.grid()
    plt.ylim((-0.001, 0.05))
    plt.title("Residual-Based resampling visualisation")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.savefig(filename)
    if verbose:
        plt.show()


def plot_reference_error(model, num_oxides, references, t_shift, loss_history,
                         verbose=False, filename="ReferenceLoss.png", plot_references=False):
    num_references = len(references)
    roi_beg, roi_end = find_region_of_main_peak(references)
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(20, 16))
    t_grid = dde.geometry.Interval(100, 800).uniform_points(2000, False)
    predicted = model.predict(t_grid)
    axs[0].plot(t_grid, predicted[:, num_oxides], label="Predicted solution")
    for i, (reference, temp) in enumerate(references[:]):
        mask = temp > 1500
        temp_shifted = temp - t_shift
        grid = np.expand_dims(temp_shifted[mask], -1)
        predicted_values_for_reference = torch.Tensor(model.predict(grid))
        error = model.data.bcs[2 * i].error(0, 0, predicted_values_for_reference.cuda(), 0, len(temp_shifted[mask]))
        mean_error = torch.mean((torch.Tensor(reference[mask]) - predicted_values_for_reference[:, num_oxides]) ** 2)
        mean_error_calculated = loss_history.loss_test[-1][2 * num_oxides + 2 * i] / (5e+0 / num_references)
        if plot_references:
            axs[0].plot(grid, reference[mask], label=f"{i + 1} reference")
        axs[0].plot(grid, np.abs(error.cpu()),
                    label=f"{i + 1} reference loss: {mean_error} and calculated loss {mean_error_calculated}")

        max_peak_grid = grid[roi_beg:roi_end + 1, :]
        predicted_values_for_main_peak = torch.Tensor(model.predict(max_peak_grid))
        main_peak_error = model.data.bcs[2 * i + 1].error(0, 0, predicted_values_for_main_peak.cuda(), 0,
                                                          len(max_peak_grid))
        axs[1].plot(max_peak_grid, main_peak_error.cpu(), label=f"{i + 1} reference main peak loss")
    axs[0].legend()
    axs[1].legend()
    axs[0].grid()
    axs[1].grid()
    axs[0].set_title("Solution and references error")
    axs[1].set_title("Solution and references error (region of main peak)")
    axs[0].set_xlabel("Time")
    axs[1].set_xlabel("Time")
    axs[0].set_ylabel("Error")
    axs[1].set_ylabel("Error")
    plt.savefig(filename)
    if verbose:
        plt.show()


def plot_ideal_functions_for_predicted_vars(trainable_variables, oxide_params, options,
                                            verbose=False, filename="IdealFunctionsForPredictedVariables.png"):
    num_oxides = len(oxide_params)
    t_shift = options["t_shift"]
    record = [elem.item() for elem in trainable_variables]
    plt.figure(figsize=(10, 6))
    t_grid = np.linspace(0, 800, 1000)
    for i, (oxide_name, oxide) in enumerate(oxide_params.items()):

        e_var = 8e+4
        k_var = e_var / oxide["Tm"] + np.log(e_var / oxide["Tm"] ** 2)
        oxide_init = create_oxide_function(k_var, e_var, 10, oxide["Tm"] - t_shift, t_shift=t_shift)
        values = oxide_init(t_grid)
        plt.plot(t_grid, values, label=f"Init {oxide_name}")

        e_var = (record[i + num_oxides]) * options["e_scale"]
        if options["direct_tmax"]:
            t_max_var = record[i] * options["tmax_scale"]
            k_var = e_var / t_max_var + np.log(e_var / t_max_var ** 2)
            t_max_var -= t_shift
        else:
            k_var = record[i] * options["k_scale"]
            oxide_ideal_non_normalised = create_oxide_function(k_var, e_var, 10, oxide["Tm"] - t_shift, t_shift=t_shift)
            values = oxide_ideal_non_normalised(t_grid)
            t_max_var = t_grid[np.argmax(values)]
        oxide_ideal = create_oxide_function(k_var, e_var, 10, t_max_var, t_shift=t_shift)
        values = oxide_ideal(t_grid)
        label = f"{oxide_name}: K = {round(k_var)}, E = {round(e_var)}, T_max = {int(t_max_var) + t_shift}"
        plt.plot(t_grid, values, label=label)

    plt.grid()
    plt.title("Ideal Functions for the predicted e and k")
    plt.legend()
    plt.savefig(filename)
    if verbose:
        plt.show()


def plot_tbeg_loss(model, oxide_params, verbose=False, filename="TbegLoss.png"):
    t_grid = dde.geometry.Interval(0, 800).uniform_points(2000, False)
    result = model.predict(t_grid)

    plt.figure(figsize=(10, 6))
    t_shift = 1400
    for i, oxide_name in enumerate(oxide_params):
        modifier = np.array(
            torch.nn.functional.sigmoid(torch.tensor(t_grid - oxide_params[oxide_name]["Tb"] + t_shift)).to("cpu"))
        plt.plot(t_grid, modifier, label=f"sigmoid for oxide: {oxide_name}")
        plt.plot(t_grid, (result[:, i:i + 1] * (1 - modifier)) ** 2,
                 label=f"initial conditions loss for oxide: {oxide_name}")

    plt.xlabel("time")
    plt.ylabel("sigmoid functions")
    plt.title("Sigmoid functions used for Temperature of Beginning incorporation")
    plt.grid()
    plt.ylim((0, 4))
    plt.legend()
    plt.savefig(filename)
    if verbose:
        plt.show()


def plot_all(model, oxide_params, references, loss_history, trainable_variables, options,
             experiment_path, verbose=False):

    t_shift = options["t_shift"]
    plot_loss_history(loss_history, verbose, os.path.join(experiment_path, "LossHistory.png"))
    plot_result(model, oxide_params, references, t_shift, verbose, os.path.join(experiment_path, "Result.png"))
    plot_ode_residual(model, oxide_params, verbose, os.path.join(experiment_path, "ODELoss.png"))
    plot_reference_error(model, len(oxide_params), references, t_shift, loss_history,
                         verbose, os.path.join(experiment_path, "ReferenceLoss.png"))
    plot_ideal_functions_for_predicted_vars(trainable_variables, oxide_params, options, verbose,
                                            os.path.join(experiment_path, "IdealFunctionsForPredictedVars.png"))
    plot_tbeg_loss(model, oxide_params, verbose, os.path.join(experiment_path, "TBegLoss.png"))
