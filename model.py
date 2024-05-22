import deepxde as dde
import numpy as np
import torch
import os
from functools import partial
from torch.nn.functional import sigmoid
from utils.preprocessing import find_region_of_main_peak, normalize_real_values, convert_grid_to_unified
from utils.callbacks import IntervalWithSmartResampling, SolutionHistory
from utils.ddeClassOverridder import ODEWithReferences, OxSepModel


def model_multiple_solutions(solution_values_array, oxides,
                             options, experiment_path,
                             trainable_vars_init=None,  pretrained_model=None):

    oxides = list(oxides.values())
    num_oxides = len(oxides)
    if trainable_vars_init is None:
        trainable_vars_init = [0] * num_oxides * 2
    get_e_functions = []
    get_k_functions = []
    get_t_max_functions = []
    k_variables = []
    e_variables = []
    t_max_variables = []
    for j, ox in enumerate(oxides):
        e_init = options["e_var_init"]
        k_init = e_init / ox["Tm"] + np.log(e_init / (ox["Tm"]) ** 2)
        get_e_functions.append(partial(options["get_e"], e_init=e_init))
        get_k_functions.append(partial(options["get_k"], k_init=k_init))
        get_t_max_functions.append(partial(options["get_t_max"], t_max_init=ox["Tm"]))
        k_variables.append(dde.Variable(trainable_vars_init[j], dtype=torch.double))
        t_max_variables.append(dde.Variable(trainable_vars_init[j], dtype=torch.double))
        e_variables.append(dde.Variable(trainable_vars_init[num_oxides + j], dtype=torch.double))

    def f(t, k, e):
        """f(t) = K - E/T(t)"""
        return k - e / options["temperature"](t)

    def df_t(t, e):
        """f'(t) = (E * T'(t)) / (T(t))^2"""
        return (e * options["temperature_derivative"](t)) / (options["temperature"](t) ** 2)

    def ode(t, v):
        """ode system: v'(t) = v(t)(df_t - exp(f(t)))"""
        oxides_odes = []
        initial_condition_checks = []
        for j, oxide in enumerate(oxides):
            oxide_function = v[:, j:j + 1]
            e_var = get_e_functions[j](e_variables[j])
            t_max_var = get_t_max_functions[j](t_max_variables[j])
            k_var = get_k_functions[j](e_var, t_max_var, k_variables[j])
            oxides_functions_derivative = dde.grad.jacobian(v, t, i=j, j=0)
            modifier = df_t(t, e_var) - torch.exp(f(t, k_var, e_var))
            direct_modifier = sigmoid(t + (oxide["Tm"] - oxide["Tb"]) - oxide["Tm"] + options["t_shift"])
            reversed_modifier = sigmoid(-t + (oxide["Tm"] - oxide["Tb"]) + oxide["Tm"] - options["t_shift"])
            region_modifier = direct_modifier * reversed_modifier
            # enhancer = oxide_function ** options["ode_loss_enhancer_power"]
            oxides_ode = (oxides_functions_derivative - oxide_function * modifier)  # * enhancer * region_modifier
            oxides_odes.append(oxides_ode)
            initial_condition_check = oxide_function * (1 - region_modifier)
            initial_condition_checks.append(initial_condition_check)

        return oxides_odes + initial_condition_checks

    def transform_output(_, v):
        oxide_functions = options["last_activation"](v)
        # for i, include_oxide in enumerate(options["oxide_toggle"]):
        #     if not include_oxide:
        #         oxide_functions[:, i] = 0
        oxide_functions_sum = torch.sum(oxide_functions, dim=-1, keepdim=True)
        return torch.concat([oxide_functions, oxide_functions_sum], dim=-1)

    def transform_input(t):
        return t / 800

    geom = dde.geometry.Interval(0, 800)

    discrete_conditions = []
    if options["normalized"]:
        references, max_value = normalize_real_values(solution_values_array)
    else:
        references = solution_values_array
        max_value = 1
    roi_beg, roi_end = find_region_of_main_peak(references)
    if options["reduced_points"]:
        references = convert_grid_to_unified(references, num_points=500)
    for solution, temp in references:
        mask = temp > options["melting_temp"]
        shifted_temp = temp - options["t_shift"]
        if options["reduced_points"]:
            discrete_conditions.append(np.expand_dims(solution[mask], -1))
        else:
            discrete_conditions.append(dde.icbc.PointSetBC(np.expand_dims(shifted_temp[mask], -1),
                                                           np.expand_dims(solution[mask], -1),
                                                           component=num_oxides))
            discrete_conditions.append(dde.icbc.PointSetBC(np.expand_dims(shifted_temp[mask][roi_beg:roi_end + 1], -1),
                                                           np.expand_dims(solution[mask][roi_beg:roi_end + 1], -1),
                                                           component=num_oxides))

    if options["reduced_points"]:
        data = ODEWithReferences(geom, ode, [], 1000, 0,
                                 train_distribution="pseudo",
                                 reference_grid=np.expand_dims(references[0][1], -1),
                                 reference_values=discrete_conditions,
                                 mpeak_beg=roi_beg, mpeak_end=roi_end)
    else:
        data = dde.data.PDE(geom, ode, discrete_conditions, 1000, 0, train_distribution="pseudo")
    if pretrained_model is not None:
        net = pretrained_model.net
    else:
        # net = dde.nn.PFNN([1] + [10 for _ in range(num_oxides)] * 3 + [num_oxides], "tanh", "Glorot uniform")
        net = dde.nn.FNN([1] + [50] * 3 + [num_oxides], ["tanh"] * 3 + ["relu"], "Glorot uniform")

    net.apply_output_transform(transform_output)
    if options["normalized_input"]:
        net.apply_feature_transform(transform_input)
    if options["normalized"]:
        model = OxSepModel(data, net, max_value)
    else:
        model = dde.Model(data, net)
    model.data.geom = IntervalWithSmartResampling(0, 800, model, num_oxides)
    model.data.train_distribution = "residual-based"
    if options["direct_tmax"]:
        external_trainable_variables = t_max_variables + e_variables
    else:
        external_trainable_variables = k_variables + e_variables
    variable = dde.callbacks.VariableValue(external_trainable_variables, period=400,
                                           filename=os.path.join(experiment_path, "variables.dat"))
    resampler = dde.callbacks.PDEPointResampler(period=1000)
    callbacks = [resampler, variable,
                 SolutionHistory(os.path.join(experiment_path, "SolutionHistory"), 0, 800, 100, period=400)]
    return model, external_trainable_variables, callbacks
