import deepxde as dde
import numpy as np
import torch
import os
from utils.preprocessing import find_region_of_main_peak
from utils.callbacks import IntervalWithSmartResampling, SolutionHistory


def model_multiple_solutions(solution_values_array, oxides, options, experiment_path, pretrained_model=None):
    oxide_names = oxides.keys()
    oxides = list(oxides.values())

    # initialize external trainable variables
    num_oxides = len(oxides)
    e_estimates = [options["e_var_init"] for _ in range(num_oxides)]
    k_estimates = [e_estimates[j] / oxides[j]["Tm"] + np.log(e_estimates[j] / (oxides[j]["Tm"]) ** 2) for j in
                   range(num_oxides)]
    k_variables = [dde.Variable(k_estimates[j] / options["k_scale"], dtype=torch.double) for j in range(num_oxides)]
    t_max_variables = [dde.Variable(oxides[j]["Tm"] / options["tmax_scale"], dtype=torch.double) for j in
                       range(num_oxides)]
    e_variables = [dde.Variable(e_estimates[j] / options["e_scale"], dtype=torch.double) for j in
                   range(num_oxides)]

    def f(t, k, e):
        """f(t) = K - E/T(t)"""
        return k - e / options["temperature"](t)

    def df_t(t, e):
        """f'(t) = (E * T'(t)) / (T(t))^2"""
        return (e * options["temperature_derivative"](t)) / (options["temperature"](t) ** 2)

    def ode(t, v):
        """ode system: v'(t) = v(t)(df_t - exp(f(t)))"""
        oxide_functions = [v[:, j:j + 1] for j in range(num_oxides)]
        e_vars = [options["get_e"](e_variable) for e_variable in e_variables]
        t_max_vars = [options["get_t_max"](t_max_variable) for t_max_variable in t_max_variables]
        k_vars = [options["get_k"](e_vars[j], t_max_vars[j], k_variables[j]) for j in range(num_oxides)]

        oxides_functions_derivatives = [dde.grad.jacobian(v, t, i=j, j=0) for j in range(num_oxides)]
        modifiers = [df_t(t, e_vars[j]) - torch.exp(f(t, k_vars[j], e_vars[j])) for j in range(num_oxides)]
        oxides_odes = [oxides_functions_derivatives[j] - oxide_functions[j] * modifiers[j] for j in range(num_oxides)]

        lengthes = [abs(oxides[j]["Tb"] - options["t_shift"]) + 1 for j in range(num_oxides)]
        modifiers = [torch.nn.functional.sigmoid(t - oxides[j]["Tb"] + options["t_shift"]) for j in range(num_oxides)]
        initial_condition_check = [(oxide_functions[j] - oxide_functions[j] * modifiers[j]) / lengthes[j] for j in
                                   range(len(oxide_functions))]

        return oxides_odes + initial_condition_check

    def transform_output(_, v):
        oxide_functions = [torch.exp(v[:, j:j + 1]) for j in range(num_oxides)]
        oxide_functions_sum = torch.zeros_like(oxide_functions[0])
        for j in range(num_oxides):
            oxide_functions_sum += oxide_functions[j]
        return torch.stack(oxide_functions + [oxide_functions_sum], axis=1).reshape(-1, num_oxides + 1)

    geom = dde.geometry.Interval(0, 800)

    discrete_conditions = []
    roi_beg, roi_end = find_region_of_main_peak(solution_values_array)
    for solution, temp in solution_values_array:
        mask = temp > options["melting_temp"]
        shifted_temp = temp - options["t_shift"]
        discrete_conditions.append(dde.icbc.PointSetBC(np.expand_dims(shifted_temp[mask], -1),
                                                       np.expand_dims(solution[mask], -1),
                                                       component=num_oxides))
        discrete_conditions.append(dde.icbc.PointSetBC(np.expand_dims(shifted_temp[mask][roi_beg:roi_end + 1], -1),
                                                       np.expand_dims(solution[mask][roi_beg:roi_end + 1], -1),
                                                       component=num_oxides))

    data = dde.data.PDE(geom, ode, discrete_conditions, 1000, 0, train_distribution="pseudo")
    if pretrained_model is not None:
        net = pretrained_model.net
    else:
        # net = dde.nn.PFNN([1] + [10 for j in range(num_oxides)] * 3 + [num_oxides], "tanh", "Glorot uniform")
        net = dde.nn.FNN([1] + [50] * 3 + [num_oxides], ["tanh"] * 3 + ["relu"], "Glorot uniform")

    net.apply_output_transform(transform_output)
    model = dde.Model(data, net)
    model.data.geom = IntervalWithSmartResampling(0, 800, model, num_oxides)
    model.data.train_distribution = "residual-based"
    if options["direct_tmax"]:
        external_trainable_variables = t_max_variables + e_variables
    else:
        external_trainable_variables = k_variables + e_variables
    variable = dde.callbacks.VariableValue(external_trainable_variables, period=50,
                                           filename=os.path.join(experiment_path, "variables.dat"))
    resampler = dde.callbacks.PDEPointResampler(period=1000)
    callbacks = [resampler, variable,
                 SolutionHistory(os.path.join(experiment_path, "SolutionHistory"), 0, 800, 100, period=50)]
    return model, external_trainable_variables, callbacks
